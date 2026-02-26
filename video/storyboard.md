# Storyboard - Think Deep, Not Just Long

## Paper Summary

| Field | Value |
|-------|-------|
| **Title** | Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens |
| **Authors / Org** | Wei-Lin Chen et al. - University of Virginia + Google |
| **Venue** | arXiv 2025 |
| **Story arc** | "Thinking longer ≠ thinking better" - token count anti-correlates with accuracy (r=-0.594), while DTR (deep-thinking ratio) positively correlates (r=0.683). The real signal is internal revision depth, not surface verbosity. |

## Tool Assignment

- **Manim**: All 6 scenes
- **Excalidraw MCP**: Scene layout sketch (completed)

---

## Scene 1: Hook (0-5s)

**Tool:** Manim
**Purpose:** Shatter the assumption that longer CoT = better reasoning.

**Title text:**
> "Does thinking longer mean thinking better?"

**Counter animation:**
- Animated decimal: 0.000 → -0.594
- Label: "correlation: token count vs accuracy"
- Color: starts WHITE, transitions to RED as it goes negative

**Visual elements:**
- Title fades in at top
- Counter centered, large font
- Subtle "X" or cross-out appears over the number

**Transition:** scene_wipe, fade all

---

## Scene 2: Setup (5-10s)

**Tool:** Manim
**Purpose:** Establish the models + benchmarks tested.

**Key elements:**
- "8 reasoning models × 4 benchmarks" subtitle
- Two rows of pill labels

**Model labels (row 1):**
- GPT-OSS-20B - color: `#4a9eed` (blue)
- GPT-OSS-120B - color: `#8b5cf6` (purple)
- DeepSeek-R1-70B - color: `#22c55e` (green)
- Qwen3-30B - color: `#f59e0b` (amber)

**Benchmark labels (row 2):**
- AIME '24 - muted pill
- AIME '25 - muted pill
- HMMT '25 - muted pill
- GPQA - muted pill

**Transition:** scene_wipe

---

## Scene 3: Method / Framework (10-16s)

**Tool:** Manim
**Purpose:** Show what DTR actually measures - prediction revision across layers.

**Diagram type:** Horizontal layer pipeline

**Content:**
- 5 rounded rectangles in a row: L₁ → L₂ → ... → Lₙ
- Below each layer: a small token prediction icon (colored dot)
- For "deep-thinking token": dots CHANGE color across layers (prediction shifts)
- For "shallow token": dots stay SAME color (prediction settled early)
- Arrow labeled "JSD" measuring the divergence
- "Deep-Thinking Ratio (DTR)" label appears at bottom

**Highlight:** The deep-thinking token row glows green, shallow token row dims

**Transition:** scene_wipe

---

## Scene 4: Results (16-22s)

**Tool:** Manim
**Purpose:** The visual punchline - correlation bars going opposite directions.

**Chart type:** Horizontal paired bars (two metrics, opposite signs)

**Data:**

| Metric | Avg Correlation | Color |
|--------|----------------|-------|
| Token Count | -0.594 | `#f85149` (red) |
| DTR | +0.683 | `#22c55e` (green) |

**Animation:**
1. Baseline (zero line) appears centered
2. Token Count bar grows LEFT (negative) - red
3. DTR bar grows RIGHT (positive) - green
4. Glow highlight on DTR bar
5. Labels: "r = -0.594" and "r = +0.683" appear at bar ends

**Secondary data (optional per-benchmark breakdown):**

| Benchmark | Token Count r | DTR r |
|-----------|--------------|-------|
| AIME '24 | -0.584 | 0.665 |
| AIME '25 | -0.608 | 0.665 |
| HMMT '25 | -0.729 | 0.735 |
| GPQA | -0.453 | 0.667 |

**Transition:** scene_wipe

---

## Scene 5: Insight (22-27s)

**Tool:** Manim
**Purpose:** Think@n - the practical application. DTR enables 50% cost reduction.

**Visualization:** Paired comparison bars

**Data:**

| Method | Accuracy | Cost (tokens) |
|--------|----------|---------------|
| Cons@n | 92.7% | 308k |
| Think@n | 94.7% | 155k |

**Animation:**
1. Two horizontal bars: Cons@n (full width = 308k) and Think@n (half width = 155k)
2. Accuracy labels appear: 92.7% vs 94.7%
3. Think@n bar glows - "higher accuracy, half the cost"

**Punchline text:**
> "Think deep, not just long."

**Transition:** scene_wipe_simple

---

## Scene 6: Closing (27-30s)

**Tool:** Manim
**Purpose:** Credits and links.

**Display:**
- Paper title (large, bold)
- "University of Virginia + Google" (subtitle, muted)
- arXiv link: `arxiv.org/abs/2602.13517`

**Fade out to black.**

---

## Production Notes

- **Target duration:** 28-30s (social media optimized)
- **Resolution:** 1920×1080 (16:9)
- **Background:** `#0d1117` (GitHub dark)
- **Font:** System default (Manim Pango)
- **Export format:** MP4 + GIF
- **No character** - text-only transitions, no mascot
