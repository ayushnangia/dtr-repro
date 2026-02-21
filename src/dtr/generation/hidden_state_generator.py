"""Hidden-state generation pipeline for DTR computation.

Provides two approaches:
- **Approach A** (:class:`HiddenStateGenerator`): Custom autoregressive loop that
  captures per-token hidden states on the fly and computes JSD/DTR incrementally.
- **Approach B** (:class:`PostHocAnalyzer`): Single forward pass over an
  already-generated sequence to (re)compute DTR metrics.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from dtr.generation.model_loader import LoadedModel
from dtr.generation.sampling import create_generator, sample_next_token
from dtr.metrics.dtr import DTRAccumulator, compute_jsd_per_layer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Container for a single generation along with all DTR-related metrics.

    Attributes
    ----------
    token_ids:
        Generated token ids (excluding the prompt).
    text:
        Decoded text of the generated tokens.
    metrics:
        Dictionary of aggregate metrics: ``dtr``, ``settling_depths``,
        ``deep_thinking_mask``, ``log_prob``, ``entropy``, and any baseline
        metrics that were computed.
    jsd_matrix:
        Optional ``(num_generated_tokens, num_layers)`` tensor of per-token,
        per-layer JSD values.  Stored only when ``store_jsd_matrix=True``
        in the generator.  Useful for sensitivity analysis.
    """

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    metrics: dict = field(default_factory=dict)
    jsd_matrix: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Approach A: Custom autoregressive loop (primary)
# ---------------------------------------------------------------------------

class HiddenStateGenerator:
    """Generate tokens one-by-one with hidden state capture for DTR computation.

    This is the primary generation strategy.  At each decoding step the full
    stack of hidden states for the *last generated position* is extracted,
    moved to CPU, and fed into a :class:`~dtr.metrics.dtr.DTRAccumulator` to
    compute JSD values incrementally.

    Parameters
    ----------
    loaded_model:
        A :class:`~dtr.generation.model_loader.LoadedModel` instance.
    max_new_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature (0 = greedy).
    top_p:
        Nucleus sampling threshold.
    seed:
        Optional random seed for reproducible sampling.
    threshold_g:
        JSD threshold *g* passed to :class:`DTRAccumulator`.
    depth_ratio_rho:
        Depth-ratio cutoff *rho* passed to :class:`DTRAccumulator`.
    store_jsd_matrix:
        If ``True``, keep the full ``(T, L)`` JSD matrix in the result
        (useful for sensitivity sweeps but costs more memory).
    """

    def __init__(
        self,
        loaded_model: LoadedModel,
        max_new_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
        threshold_g: float = 0.5,
        depth_ratio_rho: float = 0.85,
        store_jsd_matrix: bool = False,
    ) -> None:
        self.loaded_model = loaded_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.threshold_g = threshold_g
        self.depth_ratio_rho = depth_ratio_rho
        self.store_jsd_matrix = store_jsd_matrix

    # --------------------------------------------------------------------- #
    # Public API                                                              #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> GenerationResult:
        """Run custom autoregressive generation with per-token hidden-state capture.

        Parameters
        ----------
        input_ids:
            Prompt token ids, shape ``(1, seq_len)``.
        attention_mask:
            Optional attention mask, shape ``(1, seq_len)``.

        Returns
        -------
        GenerationResult
            Generated tokens, decoded text, and all DTR metrics.
        """
        model = self.loaded_model.model
        tokenizer = self.loaded_model.tokenizer
        num_layers = self.loaded_model.num_layers
        device = self.loaded_model.device

        # Move inputs to model device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Set up sampling generator
        gen = create_generator(self.seed, device=torch.device("cpu")) if self.seed is not None else None

        # Set up DTR accumulator (works on CPU to save GPU memory)
        accumulator = DTRAccumulator(
            num_layers=num_layers,
            lm_head_weight=self.loaded_model.lm_head_weight,
            layer_norm=self.loaded_model.final_layer_norm,
            threshold_g=self.threshold_g,
            depth_ratio_rho=self.depth_ratio_rho,
        )

        # Storage
        generated_ids: list[int] = []
        jsd_rows: list[torch.Tensor] = []  # each (num_layers,)
        log_probs: list[float] = []
        entropies: list[float] = []

        # Resolve EOS token id(s)
        eos_token_id = tokenizer.eos_token_id

        # KV-cache will be managed by the model via past_key_values
        past_key_values = None
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask

        _gen_start = time.monotonic()
        _log_interval = 100  # log every N tokens

        for step in range(self.max_new_tokens):
            # Forward pass with hidden states and KV cache
            outputs = model(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )

            # --- Extract hidden states for the last generated position ---
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors,
            # each of shape (batch, seq_len, hidden_dim).
            # Index 0 is the embedding output; indices 1..num_layers are the
            # transformer layer outputs.  We want layers 1..num_layers.
            hidden_states_tuple = outputs.hidden_states

            # Stack layer outputs for the last position (keep on GPU for DTR computation)
            # Shape: (num_layers, hidden_dim)
            per_layer_hs = torch.stack(
                [hidden_states_tuple[i + 1][:, -1, :].squeeze(0) for i in range(num_layers)],
                dim=0,
            ).float()

            # Feed to DTR accumulator (incremental JSD + settling)
            token_metrics = accumulator.add_token(per_layer_hs)

            if self.store_jsd_matrix:
                jsd_rows.append(token_metrics.get("jsd_vector", per_layer_hs.new_zeros(num_layers)))

            # --- Sample next token ---
            logits = outputs.logits[:, -1, :].squeeze(0).float().cpu()

            # Compute log-prob and entropy from the *unscaled* logits
            log_prob_dist = torch.log_softmax(logits, dim=-1)
            prob_dist = torch.softmax(logits, dim=-1)
            entropy = -(prob_dist * log_prob_dist).sum().item()

            next_token = sample_next_token(
                logits,
                temperature=self.temperature,
                top_p=self.top_p,
                generator=gen,
            )

            log_probs.append(log_prob_dist[next_token].item())
            entropies.append(entropy)
            generated_ids.append(next_token)

            # --- Periodic progress logging ---
            if (step + 1) % _log_interval == 0:
                elapsed = time.monotonic() - _gen_start
                tok_per_sec = (step + 1) / elapsed
                logger.info(
                    "  token %d | %.1f tok/s | settling_depth=%d | is_deep=%s",
                    step + 1, tok_per_sec,
                    token_metrics["settling_depth"], token_metrics["is_deep"],
                )

            # --- Clean up GPU tensors we no longer need ---
            del hidden_states_tuple, per_layer_hs, logits, outputs
            # (past_key_values is kept for the next step)

            # --- Check stopping condition ---
            if next_token == eos_token_id:
                break

            # --- Prepare next step ---
            past_key_values = model._reorder_cache(
                past_key_values, torch.tensor([0], device=device)
            ) if hasattr(model, "_reorder_cache") else past_key_values

            # For the next step we only feed the newly generated token
            cur_input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)

            # Extend attention mask by one position
            if cur_attention_mask is not None:
                cur_attention_mask = torch.cat(
                    [cur_attention_mask, torch.ones((1, 1), dtype=cur_attention_mask.dtype, device=device)],
                    dim=1,
                )

        # --- Aggregate results ---
        dtr_results = accumulator.get_results()

        metrics: dict = {
            "dtr": dtr_results["dtr"],
            "settling_depths": dtr_results["settling_depths"],
            "deep_thinking_mask": dtr_results["deep_thinking_mask"],
            "mean_log_prob": sum(log_probs) / len(log_probs) if log_probs else 0.0,
            "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "num_generated_tokens": len(generated_ids),
            "log_probs": log_probs,
            "entropies": entropies,
        }

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        jsd_matrix = torch.stack(jsd_rows, dim=0) if (self.store_jsd_matrix and jsd_rows) else None

        logger.info(
            "Generation complete: %d tokens, DTR=%.4f",
            len(generated_ids),
            metrics["dtr"],
        )

        return GenerationResult(
            token_ids=generated_ids,
            text=text,
            metrics=metrics,
            jsd_matrix=jsd_matrix,
        )


# ---------------------------------------------------------------------------
# Approach B: Post-hoc forward pass (fallback)
# ---------------------------------------------------------------------------

class PostHocAnalyzer:
    """Compute DTR from an already-generated sequence via a single forward pass.

    This is the fallback/reanalysis strategy.  Given a prompt and a completed
    generation, it runs one forward pass through the model, extracts all hidden
    states, and computes DTR metrics.  This is more memory-efficient for
    re-computing DTR with different ``g``/``rho`` settings because it avoids
    storing KV caches.

    Parameters
    ----------
    loaded_model:
        A :class:`~dtr.generation.model_loader.LoadedModel` instance.
    chunk_size:
        Maximum number of tokens to process at once during the forward pass.
        Longer sequences are split into chunks to limit peak GPU memory.
    """

    def __init__(self, loaded_model: LoadedModel, chunk_size: int = 2048) -> None:
        self.loaded_model = loaded_model
        self.chunk_size = chunk_size

    @torch.no_grad()
    def analyze(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        threshold_g: float = 0.5,
        depth_ratio_rho: float = 0.85,
        use_cpu_dtr: bool = False,
    ) -> dict:
        """Run a forward pass on the concatenated sequence and compute DTR.

        Parameters
        ----------
        input_ids:
            Prompt token ids, shape ``(1, prompt_len)`` or ``(prompt_len,)``.
        generated_ids:
            Generated token ids, shape ``(1, gen_len)`` or ``(gen_len,)``.
        threshold_g:
            JSD threshold for deep-thinking classification.
        depth_ratio_rho:
            Depth-ratio cutoff for settling-depth computation.
        use_cpu_dtr:
            If ``True``, use a CPU copy of the final layer norm for DTR
            computation.  This avoids moving hidden states back to GPU and
            keeps the GPU free for the next generation.

        Returns
        -------
        dict
            Dictionary with keys: ``dtr``, ``settling_depths``,
            ``deep_thinking_mask``, ``jsd_matrix`` (``(gen_len, num_layers)``).
        """
        model = self.loaded_model.model
        num_layers = self.loaded_model.num_layers
        device = self.loaded_model.device

        # Flatten to 1-D if needed
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        if generated_ids.dim() == 2:
            generated_ids = generated_ids.squeeze(0)

        prompt_len = input_ids.shape[0]
        gen_len = generated_ids.shape[0]

        # Concatenate prompt + generation
        full_ids = torch.cat([input_ids, generated_ids], dim=0).unsqueeze(0).to(device)
        total_len = full_ids.shape[1]

        # Process in chunks to manage memory
        all_hidden_states = self._forward_chunked(model, full_ids, num_layers, total_len)

        # all_hidden_states: list of num_layers tensors, each (total_len, hidden_dim) on CPU
        # We only need the generated positions (indices prompt_len .. total_len-1)
        # Stack into (gen_len, num_layers, hidden_dim)
        gen_hidden = torch.stack(
            [layer_hs[prompt_len:prompt_len + gen_len] for layer_hs in all_hidden_states],
            dim=1,
        )  # (gen_len, num_layers, hidden_dim)

        # Build DTR accumulator and process each token
        layer_norm = (
            self.loaded_model.cpu_final_layer_norm if use_cpu_dtr
            else self.loaded_model.final_layer_norm
        )
        accumulator = DTRAccumulator(
            num_layers=num_layers,
            lm_head_weight=self.loaded_model.lm_head_weight,
            layer_norm=layer_norm,
            threshold_g=threshold_g,
            depth_ratio_rho=depth_ratio_rho,
        )

        jsd_rows: list[torch.Tensor] = []
        for t in range(gen_len):
            per_token_hs = gen_hidden[t]  # (num_layers, hidden_dim)
            token_metrics = accumulator.add_token(per_token_hs)
            jsd_vec = token_metrics.get("jsd_vector", per_token_hs.new_zeros(num_layers))
            jsd_rows.append(jsd_vec)

        results = accumulator.get_results()
        jsd_matrix = torch.stack(jsd_rows, dim=0) if jsd_rows else None  # (gen_len, num_layers)

        return {
            "dtr": results["dtr"],
            "settling_depths": results["settling_depths"],
            "deep_thinking_mask": results["deep_thinking_mask"],
            "jsd_matrix": jsd_matrix,
        }

    def _forward_chunked(
        self,
        model: torch.nn.Module,
        full_ids: torch.Tensor,
        num_layers: int,
        total_len: int,
    ) -> list[torch.Tensor]:
        """Run forward pass, optionally chunked, returning per-layer hidden states on CPU.

        Parameters
        ----------
        model:
            The causal-LM model.
        full_ids:
            Token ids, shape ``(1, total_len)``.
        num_layers:
            Number of transformer layers expected.
        total_len:
            Total sequence length.

        Returns
        -------
        list[torch.Tensor]
            List of *num_layers* tensors, each ``(total_len, hidden_dim)`` on CPU.
        """
        device = full_ids.device

        if total_len <= self.chunk_size:
            # Single forward pass
            outputs = model(
                input_ids=full_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            # hidden_states: tuple of (num_layers+1) tensors, each (1, total_len, hidden_dim)
            hidden_states = outputs.hidden_states
            result = [
                hidden_states[i + 1].squeeze(0).cpu().float()
                for i in range(num_layers)
            ]
            del outputs, hidden_states
            return result

        # Chunked forward pass for long sequences
        # We process chunks sequentially, using KV cache to carry context forward.
        layer_chunks: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        past_key_values = None

        for start in range(0, total_len, self.chunk_size):
            end = min(start + self.chunk_size, total_len)
            chunk_ids = full_ids[:, start:end]

            outputs = model(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )

            hidden_states = outputs.hidden_states
            for layer_idx in range(num_layers):
                # (1, chunk_len, hidden_dim) -> (chunk_len, hidden_dim)
                chunk_hs = hidden_states[layer_idx + 1].squeeze(0).cpu().float()
                layer_chunks[layer_idx].append(chunk_hs)

            past_key_values = outputs.past_key_values
            del outputs, hidden_states

        # Concatenate chunks per layer
        result = [torch.cat(chunks, dim=0) for chunks in layer_chunks]
        return result


# ---------------------------------------------------------------------------
# Post-hoc generation: fast generate then single forward pass for DTR
# ---------------------------------------------------------------------------

class PostHocGenerator:
    """Generate tokens at full speed via ``model.generate()``, then compute DTR
    from a single post-hoc forward pass with hidden states offloaded to CPU.

    This trades exact per-token streaming metrics for significantly higher
    throughput on memory-constrained devices (e.g. MIG slices).

    Parameters
    ----------
    loaded_model:
        A :class:`~dtr.generation.model_loader.LoadedModel` instance.
    max_new_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature.
    top_p:
        Nucleus sampling threshold.
    seed:
        Random seed for reproducibility.  Applied as a *global*
        ``torch.manual_seed`` (unlike ``HiddenStateGenerator`` which uses a
        per-Generator ``torch.Generator``), so token sequences will differ
        between modes even with the same seed value.
    threshold_g:
        JSD threshold *g* passed to :class:`DTRAccumulator`.
    depth_ratio_rho:
        Depth-ratio cutoff *rho* passed to :class:`DTRAccumulator`.
    store_jsd_matrix:
        If ``True``, keep the full ``(T, L)`` JSD matrix in the result.
    chunk_size:
        Maximum tokens per forward-pass chunk in the post-hoc analysis.
    """

    def __init__(
        self,
        loaded_model: LoadedModel,
        max_new_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
        threshold_g: float = 0.5,
        depth_ratio_rho: float = 0.85,
        store_jsd_matrix: bool = False,
        chunk_size: int = 2048,
    ) -> None:
        self.loaded_model = loaded_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.threshold_g = threshold_g
        self.depth_ratio_rho = depth_ratio_rho
        self.store_jsd_matrix = store_jsd_matrix
        self.chunk_size = chunk_size

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> GenerationResult:
        """Generate tokens with ``model.generate()`` then compute DTR post-hoc.

        Parameters
        ----------
        input_ids:
            Prompt token ids, shape ``(1, seq_len)``.
        attention_mask:
            Optional attention mask, shape ``(1, seq_len)``.

        Returns
        -------
        GenerationResult
            Generated tokens, decoded text, and all DTR metrics.
        """
        model = self.loaded_model.model
        tokenizer = self.loaded_model.tokenizer
        device = self.loaded_model.device

        # Move inputs to model device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        prompt_len = input_ids.shape[1]

        # ----- Phase 1: Fast generation via model.generate() ----- #
        if self.seed is not None:
            torch.manual_seed(self.seed)

        _gen_start = time.monotonic()

        gen_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            output_logits=True,
            return_dict_in_generate=True,
        )

        _gen_elapsed = time.monotonic() - _gen_start

        # Extract generated token ids (exclude prompt)
        full_sequence = gen_output.sequences[0]  # (prompt_len + gen_len,)
        generated_ids = full_sequence[prompt_len:].tolist()

        # Extract log-probs and entropies from raw (unprocessed) logits,
        # matching what streaming mode computes from outputs.logits.
        log_probs: list[float] = []
        entropies: list[float] = []
        for step_logits in gen_output.logits:
            # step_logits: (batch=1, vocab_size) — raw model logits
            logits = step_logits[0].float().cpu()
            log_prob_dist = torch.log_softmax(logits, dim=-1)
            prob_dist = torch.softmax(logits, dim=-1)
            entropy = -(prob_dist * log_prob_dist).sum().item()
            # Log-prob of the token that was actually sampled
            token_id = generated_ids[len(log_probs)]
            log_probs.append(log_prob_dist[token_id].item())
            entropies.append(entropy)

        num_gen = len(generated_ids)
        gen_tok_per_sec = num_gen / _gen_elapsed if _gen_elapsed > 0 else 0
        logger.info(
            "PostHoc phase 1 (generate): %d tokens in %.1fs (%.1f tok/s)",
            num_gen, _gen_elapsed, gen_tok_per_sec,
        )

        # Free generation output to reclaim GPU memory before forward pass
        del gen_output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ----- Phase 2: Post-hoc DTR computation ----- #
        _dtr_start = time.monotonic()

        analyzer = PostHocAnalyzer(self.loaded_model, chunk_size=self.chunk_size)
        dtr_results = analyzer.analyze(
            input_ids=input_ids.squeeze(0).cpu(),
            generated_ids=torch.tensor(generated_ids, dtype=torch.long),
            threshold_g=self.threshold_g,
            depth_ratio_rho=self.depth_ratio_rho,
            use_cpu_dtr=True,
        )

        _dtr_elapsed = time.monotonic() - _dtr_start
        logger.info(
            "PostHoc phase 2 (DTR): %.1fs for %d tokens",
            _dtr_elapsed, num_gen,
        )

        # ----- Assemble result ----- #
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        metrics: dict = {
            "dtr": dtr_results["dtr"],
            "settling_depths": dtr_results["settling_depths"],
            "deep_thinking_mask": dtr_results["deep_thinking_mask"],
            "mean_log_prob": sum(log_probs) / len(log_probs) if log_probs else 0.0,
            "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "num_generated_tokens": num_gen,
            "log_probs": log_probs,
            "entropies": entropies,
        }

        jsd_matrix = dtr_results["jsd_matrix"] if self.store_jsd_matrix else None

        logger.info(
            "PostHoc generation complete: %d tokens, DTR=%.4f (gen=%.1fs, dtr=%.1fs)",
            num_gen, metrics["dtr"], _gen_elapsed, _dtr_elapsed,
        )

        return GenerationResult(
            token_ids=generated_ids,
            text=text,
            metrics=metrics,
            jsd_matrix=jsd_matrix,
        )
