# DTR Distance Metrics: Alternatives to Full-Vocab JSD

The baseline DTR computation projects hidden states through the full unembedding
matrix (151K vocab) and computes JSD. This is the bottleneck -- O(d * V * L) per
token where V = 151,936.

## What works

### 1. SVD-compressed unembedding + JSD (recommended)

Pre-compute the top-k SVD of W_unembed once. Project hidden states to k dims
instead of V dims. Then compute JSD in the compressed space.

- **Cost**: O(d * k * L) per token, where k = 256-512
- **Speedup**: ~300-600x over full vocab
- **Quality**: Captures dominant modes of distributional variation. For settling
  depth (a coarse question), this is likely sufficient.
- **Theory**: Top singular components of the unembedding capture the most
  variance in logit space.

### 2. Fixed-set top-k JSD

Pick the top-k token indices (from final layer or a calibration set). Only
multiply hidden states by those k rows of W_unembed.

- **Cost**: O(d * k * L) per token
- **Quality**: "Has the top-100 prediction settled?" -- semantically clear
- **Downside**: Requires choosing which tokens to track; misses tail shifts

### 3. Cosine distance (already implemented)

Just `1 - cos_sim(h_l, h_L)` in hidden space. No vocab projection.

- **Cost**: O(d * L) -- trivially cheap
- **What it misses**: Norm changes, unembedding structure. Two vectors can have
  cosine ~1.0 but produce very different softmax distributions.
- **Good for**: Quick sanity check, identifying obviously-unsettled layers

### 4. Norm-weighted cosine (hybrid)

`d(h_i, h_j) = (1 - cos(h_i, h_j)) * (||h_i|| + ||h_j||) / 2`

Captures both direction and scale. Cheap (O(d)), more informative than plain
cosine. Heuristic but sensible.

## What doesn't work

| Method | Why not |
|--------|---------|
| CKA | Per-dataset metric, not per-token. Degenerates for single vectors. |
| MMD (RBF kernel) | Reduces to L2 for single vectors. No advantage. |
| Energy distance | Same -- reduces to 2 * L2 for single vectors. |
| Wasserstein on vocab | O(V^2) -- more expensive than JSD, not less. |
| Random projection of hidden states | Doesn't approximate the unembedding; random directions are meaningless. |
| Top-k JSD on full logits | Still needs O(d * V) projection first. Doesn't save the bottleneck. |

## Summary

| Method | Cost/layer | Distributional? | Hidden-space only? |
|--------|-----------|-----------------|-------------------|
| Full JSD (baseline) | O(dV) | Yes | No |
| SVD-compressed JSD | O(dk) | Mostly | No (uses W_u) |
| Fixed top-k JSD | O(dk) | Mostly | No (uses W_u) |
| Cosine | O(d) | No | Yes |
| Norm-weighted cosine | O(d) | No | Yes |
