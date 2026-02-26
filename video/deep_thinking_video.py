"""Deep-Thinking Tokens - Paper Explainer Video
Manim Community Edition v0.19+

Paper: "Think Deep, Not Just Long: Measuring LLM Reasoning Effort
        via Deep-Thinking Tokens"
Authors: Wei-Lin Chen et al. (University of Virginia + Google)

Render:
  Test:   manim -ql deep_thinking_video.py DeepThinkingVideo
  Final:  manim -qh deep_thinking_video.py DeepThinkingVideo
  4K:     manim -qk deep_thinking_video.py DeepThinkingVideo
"""

from manim import *
import numpy as np

# ── Palette (GitHub Dark Theme) ───────────────────────────────────────
BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#30363d"
WHITE   = "#e6edf3"
MUTED   = "#8b949e"
ACCENT  = "#58a6ff"
YELLOW  = "#f0c040"
RED     = "#f85149"
GREEN   = "#2ecc71"

# ── Model Colors ─────────────────────────────────────────────────────
C_OSS_20B    = "#4a9eed"   # blue
C_OSS_120B   = "#8b5cf6"   # purple
C_DEEPSEEK   = "#22c55e"   # green
C_QWEN       = "#f59e0b"   # amber

MODEL_COLORS = {
    "GPT-OSS-20B":      C_OSS_20B,
    "GPT-OSS-120B":     C_OSS_120B,
    "DeepSeek-R1-70B":  C_DEEPSEEK,
    "Qwen3-30B":        C_QWEN,
}

# ── Correlation Data (from Table 1, averaged over 8 model variants) ──
CORR_DATA = [
    ("Token Length",     -0.594, RED),
    ("Rev. Token Len.",   0.594, MUTED),
    ("Log Probability",   0.527, MUTED),
    ("Neg. Perplexity",   0.219, MUTED),
    ("Neg. Entropy",      0.571, MUTED),
    ("Self-Certainty",    0.605, MUTED),
    ("DTR (Ours)",        0.683, GREEN),
]

# ── Per-benchmark correlation (Token Count vs DTR) ───────────────────
BENCHMARK_CORR = [
    ("AIME '25",  -0.608, 0.665),
    ("AIME '24",  -0.584, 0.665),
    ("HMMT '25",  -0.729, 0.735),
    ("GPQA-D",    -0.453, 0.667),
]

# ── Think@n data (OSS-120B-medium, AIME 2025) ───────────────────────
THINK_DATA = [
    ("Cons@n",    92.7, 307.6, MUTED),
    ("Short@n",   87.3, 255.7, MUTED),
    ("Self-Cert@n", 87.3, 150.6, ACCENT),
    ("Think@n",   94.7, 155.4, GREEN),
]


class DeepThinkingVideo(Scene):
    """Animated explainer for the Deep-Thinking Tokens paper."""

    FONT = "Avenir Next"

    def construct(self):
        self.camera.background_color = BG
        self.character = None
        self.logo = None

        self.scene_1_hook()
        self.scene_wipe()
        self.scene_2_setup()
        self.scene_wipe()
        self.scene_3_method()
        self.scene_wipe()
        self.scene_4_results()
        self.scene_wipe()
        self.scene_5_insight()
        self.scene_wipe_simple()
        self.scene_6_closing()

    # ── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def txt(s, font_size=24, color=WHITE, weight=NORMAL, **kw):
        """Render text with correct kerning (Pango scale trick).

        The multiplier is capped so that (font_size * mult * char_count)
        doesn't exceed Pango's internal line-wrap width. Without this,
        long strings at large rendered sizes wrap prematurely and overlap
        when scaled back down.
        """
        # Estimate rendered pixel width: ~0.7 * font_size * mult * len
        # Pango wraps around ~3000px; bold/wide fonts need extra margin
        longest_line = max(s.split("\n"), key=len)
        char_w_factor = 0.70
        max_render_px = 2800
        max_mult_for_width = max_render_px / (char_w_factor * font_size * max(len(longest_line), 1))
        mult = min(10, 400 / max(font_size, 1), max_mult_for_width)
        mult = max(mult, 1)  # never go below 1x
        t = Text(s, font=DeepThinkingVideo.FONT,
                 font_size=round(font_size * mult),
                 color=color, weight=weight, **kw)
        t.scale(1 / mult)
        return t

    def pill(self, text, color, w=2.4, h=0.48, fs=20):
        """Pill-shaped label - frosted glass effect."""
        rect = RoundedRectangle(
            width=w, height=h, corner_radius=h / 2,
            fill_color=color, fill_opacity=0.22,
            stroke_color=color, stroke_width=1.5, stroke_opacity=0.9,
        )
        label = self.txt(text, font_size=fs, color=WHITE)
        return VGroup(rect, label)

    def scene_wipe(self, target_pos=ORIGIN):
        """Fade out everything except persistent elements."""
        keep = set()
        if self.character:
            keep.add(self.character)
        if self.logo:
            keep.add(self.logo)
        others = [m for m in self.mobjects if m not in keep]
        fade_anims = [FadeOut(m, shift=DOWN * 0.2) for m in others]
        if fade_anims:
            self.play(*fade_anims, run_time=0.5)

    def scene_wipe_simple(self, target_pos=ORIGIN):
        """Simple wipe - fade everything."""
        keep = set()
        if self.character:
            keep.add(self.character)
        if self.logo:
            keep.add(self.logo)
        others = [m for m in self.mobjects if m not in keep]
        fade_anims = [FadeOut(m, shift=DOWN * 0.2) for m in others]
        if fade_anims:
            self.play(
                *fade_anims,
                rate_func=rate_functions.ease_in_out_cubic,
                run_time=0.5,
            )

    def glow_highlight(self, mobject, color=YELLOW):
        """Glow highlight around a mobject."""
        outer = RoundedRectangle(
            width=mobject.width + 0.22, height=mobject.height + 0.22,
            corner_radius=0.06,
            stroke_color=color, stroke_width=4, stroke_opacity=0.35,
            fill_color=color, fill_opacity=0.06,
        ).move_to(mobject.get_center())
        inner = RoundedRectangle(
            width=mobject.width + 0.06, height=mobject.height + 0.06,
            corner_radius=0.04,
            stroke_color=color, stroke_width=2.5, stroke_opacity=0.9,
            fill_opacity=0,
        ).move_to(mobject.get_center())
        return VGroup(outer, inner)

    # ── Scene 1: Hook ────────────────────────────────────────────────

    def scene_1_hook(self):
        """Provocative question + negative correlation reveal."""
        # Three short lines to avoid Pango wrapping at large sizes
        q1 = self.txt("Does thinking longer", font_size=42, weight=BOLD)
        q2 = self.txt("mean thinking better?", font_size=42, weight=BOLD)
        question = VGroup(q1, q2).arrange(DOWN, buff=0.12, center=True)
        question.move_to(UP * 1.8)

        self.play(FadeIn(question, shift=DOWN * 0.2), run_time=0.6)
        self.wait(0.4)

        # Animated correlation counter
        num = DecimalNumber(
            0, num_decimal_places=3, font_size=96,
            color=WHITE, include_sign=True,
        )
        num.move_to(DOWN * 0.3)

        label = self.txt(
            "correlation: token count vs accuracy",
            font_size=22, color=MUTED,
        )
        label.next_to(num, DOWN, buff=0.25)

        self.play(FadeIn(num), FadeIn(label, shift=UP * 0.1), run_time=0.3)

        self.play(
            ChangeDecimalToValue(num, -0.594,
                                 rate_func=rate_functions.ease_out_cubic),
            num.animate.set_color(RED),
            run_time=1.2,
        )

        cross = VGroup(
            Line(UP * 0.4 + LEFT * 0.4, DOWN * 0.4 + RIGHT * 0.4,
                 color=RED, stroke_width=4),
            Line(UP * 0.4 + RIGHT * 0.4, DOWN * 0.4 + LEFT * 0.4,
                 color=RED, stroke_width=4),
        ).next_to(num, RIGHT, buff=0.3)
        self.play(Create(cross), run_time=0.3)

        self.wait(0.6)

    # ── Scene 2: Setup ───────────────────────────────────────────────

    def scene_2_setup(self):
        """Models and benchmarks tested."""
        h1 = self.txt("8 model variants", font_size=26, color=MUTED)
        hx = self.txt("x", font_size=26, color=MUTED)
        h2 = self.txt("4 benchmarks", font_size=26, color=MUTED)
        heading = VGroup(h1, hx, h2).arrange(RIGHT, buff=0.25)
        heading.move_to(UP * 3.0)
        self.play(FadeIn(heading), run_time=0.4)

        # Model pills
        model_pills = VGroup(*[
            self.pill(name, color, w=2.8, h=0.45, fs=17)
            for name, color in MODEL_COLORS.items()
        ]).arrange(RIGHT, buff=0.2).move_to(UP * 1.5)

        self.play(
            LaggedStart(
                *[FadeIn(p, shift=UP * 0.15) for p in model_pills],
                lag_ratio=0.1,
            ),
            run_time=0.8,
        )

        # Benchmark pills
        benchmarks = ["AIME '24", "AIME '25", "HMMT '25", "GPQA-D"]
        bench_pills = VGroup(*[
            self.pill(b, BORDER, w=2.0, h=0.40, fs=16)
            for b in benchmarks
        ]).arrange(RIGHT, buff=0.2).move_to(DOWN * 0.2)

        bench_label = self.txt("Benchmarks", font_size=20, color=MUTED)
        bench_label.next_to(bench_pills, UP, buff=0.2)

        self.play(
            FadeIn(bench_label),
            LaggedStart(
                *[FadeIn(p, shift=UP * 0.12) for p in bench_pills],
                lag_ratio=0.1,
            ),
            run_time=0.7,
        )

        key_q = self.txt(
            "Which metric best predicts reasoning success?",
            font_size=22, color=YELLOW,
        )
        key_q.move_to(DOWN * 1.8)
        self.play(FadeIn(key_q, shift=UP * 0.1), run_time=0.5)
        self.wait(0.6)

    # ── Scene 3: Method ──────────────────────────────────────────────

    def scene_3_method(self):
        """DTR concept - prediction revision across layers."""
        ht = self.txt("Deep-Thinking Ratio", font_size=32,
                      color=WHITE, weight=BOLD)
        hs = self.txt("(DTR)", font_size=24, color=MUTED)
        heading = VGroup(ht, hs).arrange(RIGHT, buff=0.2,
                                          aligned_edge=DOWN)
        heading.move_to(UP * 3.2)
        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=0.5)

        # Layer pipeline
        layer_labels = ["L1", "L2", "L3", "...", "Ln"]
        layers = VGroup()
        for lbl in layer_labels:
            box = RoundedRectangle(
                width=1.1, height=0.7, corner_radius=0.1,
                fill_color=SURFACE, fill_opacity=0.9,
                stroke_color=ACCENT, stroke_width=1.5,
            )
            text = self.txt(lbl, font_size=22, color=ACCENT)
            layers.add(VGroup(box, text))
        layers.arrange(RIGHT, buff=0.5).move_to(UP * 1.2)

        arrows = VGroup()
        for i in range(len(layers) - 1):
            arr = Arrow(
                layers[i].get_right(), layers[i + 1].get_left(),
                buff=0.08, color=BORDER, stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            )
            arrows.add(arr)

        self.play(
            LaggedStart(*[FadeIn(l, shift=RIGHT * 0.1) for l in layers],
                        lag_ratio=0.08),
            run_time=0.6,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.08),
            run_time=0.4,
        )

        # Row labels - consistent left edge
        ROW_LABEL_X = -5.8
        DEEP_Y = -0.3
        SHALLOW_Y = -1.7

        dt_label = self.txt("Deep token:", font_size=18, color=GREEN)
        dt_label.set_y(DEEP_Y)
        dt_label.align_to(np.array([ROW_LABEL_X, 0, 0]), LEFT)

        st_label = self.txt("Shallow token:", font_size=18, color=RED)
        st_label.set_y(SHALLOW_Y)
        st_label.align_to(np.array([ROW_LABEL_X, 0, 0]), LEFT)

        # Deep-thinking token dots (predictions CHANGE color)
        dt_dots = VGroup()
        dot_colors_deep = [RED, RED, YELLOW, YELLOW, GREEN]
        for i, layer in enumerate(layers):
            dot = Dot(
                point=layer.get_center() + DOWN * 1.1,
                radius=0.15,
                color=dot_colors_deep[i],
                fill_opacity=0.9,
            )
            dt_dots.add(dot)

        # Shallow token dots (predictions STAY same)
        st_dots = VGroup()
        for i, layer in enumerate(layers):
            dot = Dot(
                point=layer.get_center() + DOWN * 2.5,
                radius=0.15,
                color=ACCENT,
                fill_opacity=0.4,
            )
            st_dots.add(dot)

        self.play(
            FadeIn(dt_label), FadeIn(st_label),
            LaggedStart(*[GrowFromCenter(d) for d in dt_dots], lag_ratio=0.1),
            LaggedStart(*[GrowFromCenter(d) for d in st_dots], lag_ratio=0.1),
            run_time=0.8,
        )

        change_label = self.txt("predictions shift", font_size=16, color=GREEN)
        change_label.next_to(dt_dots, RIGHT, buff=0.3)

        same_label = self.txt("settled early", font_size=16, color=MUTED)
        same_label.next_to(st_dots, RIGHT, buff=0.3)

        self.play(
            FadeIn(change_label, shift=LEFT * 0.1),
            FadeIn(same_label, shift=LEFT * 0.1),
            run_time=0.4,
        )

        formula = self.txt(
            "DTR = fraction of deep-thinking tokens",
            font_size=22, color=YELLOW,
        )
        formula.move_to(DOWN * 3.2)
        self.play(FadeIn(formula, shift=UP * 0.1), run_time=0.4)
        self.wait(0.8)

    # ── Scene 4: Results ─────────────────────────────────────────────

    def scene_4_results(self):
        """Correlation comparison - the visual punchline."""
        # Two-line heading to avoid wrapping
        h1 = self.txt("Correlation with Accuracy", font_size=24, color=MUTED)
        h2 = self.txt("(avg across all models)", font_size=20, color=MUTED)
        heading = VGroup(h1, h2).arrange(DOWN, buff=0.08, center=True)
        heading.move_to(UP * 3.3)
        self.play(FadeIn(heading), run_time=0.4)

        # ── Layout ──
        BAR_H      = 0.36
        ROW_GAP    = 0.14
        MAX_W      = 3.8
        ZERO_X     = 0.5
        TOP_Y      = 2.1
        LABEL_LEFT = -6.2

        n = len(CORR_DATA)
        bottom_y = TOP_Y - (n - 1) * (BAR_H + ROW_GAP)

        # Zero line
        zero_line = DashedLine(
            start=np.array([ZERO_X, TOP_Y + 0.3, 0]),
            end=np.array([ZERO_X, bottom_y - 0.4, 0]),
            color=BORDER, stroke_width=1.5, dash_length=0.1,
        )
        zero_label = self.txt("r = 0", font_size=14, color=MUTED)
        zero_label.next_to(zero_line, UP, buff=0.05)

        self.play(Create(zero_line), FadeIn(zero_label), run_time=0.3)

        bars = VGroup()
        labels = VGroup()
        values = VGroup()

        for i, (name, r_val, col) in enumerate(CORR_DATA):
            y = TOP_Y - i * (BAR_H + ROW_GAP)
            bar_w = abs(r_val) * MAX_W
            is_highlight = col != MUTED

            bar = Rectangle(
                width=bar_w, height=BAR_H,
                fill_color=col, fill_opacity=0.88, stroke_width=0,
            )
            if r_val < 0:
                bar.move_to(np.array([ZERO_X - bar_w / 2, y, 0]))
            else:
                bar.move_to(np.array([ZERO_X + bar_w / 2, y, 0]))

            lbl = self.txt(
                name, font_size=15,
                color=WHITE if is_highlight else MUTED,
                weight=BOLD if is_highlight else NORMAL,
            )
            lbl.set_y(y)
            lbl.align_to(np.array([LABEL_LEFT, 0, 0]), LEFT)

            vtxt = self.txt(
                f"{r_val:+.3f}", font_size=14,
                color=WHITE if is_highlight else MUTED,
                weight=BOLD if is_highlight else NORMAL,
            )
            if r_val < 0:
                vtxt.next_to(bar, LEFT, buff=0.1)
            else:
                vtxt.next_to(bar, RIGHT, buff=0.1)

            bars.add(bar)
            labels.add(lbl)
            values.add(vtxt)

        self.play(
            LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.05),
            run_time=0.4,
        )
        self.play(
            LaggedStart(
                *[GrowFromEdge(b, LEFT if CORR_DATA[i][1] >= 0 else RIGHT)
                  for i, b in enumerate(bars)],
                lag_ratio=0.07,
            ),
            LaggedStart(
                *[FadeIn(v, shift=LEFT * 0.05 if CORR_DATA[i][1] < 0 else RIGHT * 0.05)
                  for i, v in enumerate(values)],
                lag_ratio=0.07,
            ),
            run_time=1.4,
            rate_func=rate_functions.ease_out_cubic,
        )

        glow_dtr = self.glow_highlight(bars[-1], color=GREEN)
        glow_tok = self.glow_highlight(bars[0], color=RED)
        self.play(FadeIn(glow_dtr), FadeIn(glow_tok), run_time=0.4)
        self.wait(0.8)

    # ── Scene 5: Insight ─────────────────────────────────────────────

    def scene_5_insight(self):
        """Think@n - the practical payoff."""
        heading = self.txt("Think@n: Select by DTR, Not Length",
                           font_size=26, color=WHITE, weight=BOLD)
        heading.move_to(UP * 3.2)
        self.play(FadeIn(heading, shift=DOWN * 0.15), run_time=0.5)

        sub = self.txt("OSS-120B-medium | AIME 2025 | n=32",
                       font_size=16, color=MUTED)
        sub.next_to(heading, DOWN, buff=0.12)
        self.play(FadeIn(sub), run_time=0.3)

        # ── Layout ──
        LABEL_LEFT = -5.8
        BAR_LEFT   = -3.2
        BAR_MAX_W  = 7.0
        BAR_H      = 0.50
        ROW_GAP    = 0.30

        methods = [
            ("Cons@n",  92.7, 307.6, MUTED),
            ("Think@n", 94.7, 155.4, GREEN),
        ]

        def make_bar_row(name, value, max_val, unit, col, y):
            mlbl = self.txt(name, font_size=18, color=WHITE, weight=BOLD)
            mlbl.set_y(y)
            mlbl.align_to(np.array([LABEL_LEFT, 0, 0]), LEFT)

            bar_w = (value / max_val) * BAR_MAX_W
            bar = RoundedRectangle(
                width=bar_w, height=BAR_H, corner_radius=0.06,
                fill_color=col, fill_opacity=0.85, stroke_width=0,
            )
            bar.move_to(np.array([BAR_LEFT + bar_w / 2, y, 0]))

            val = self.txt(f"{value}{unit}", font_size=19,
                           color=WHITE, weight=BOLD)
            val.next_to(bar, RIGHT, buff=0.12)

            return mlbl, bar, val

        # ── Accuracy ──
        acc_title = self.txt("Accuracy", font_size=20,
                             color=MUTED, weight=BOLD)
        acc_title.set_y(1.8)
        acc_title.align_to(np.array([LABEL_LEFT, 0, 0]), LEFT)
        self.play(FadeIn(acc_title), run_time=0.2)

        acc_items = []
        for i, (name, acc, cost, col) in enumerate(methods):
            y = 1.2 - i * (BAR_H + ROW_GAP)
            acc_items.append(make_bar_row(name, acc, 100, "%", col, y))

        acc_bars = VGroup(*[item[1] for item in acc_items])
        acc_extras = VGroup(*[VGroup(item[0], item[2]) for item in acc_items])

        self.play(
            *[FadeIn(e) for e in acc_extras],
            LaggedStart(
                *[GrowFromEdge(b, LEFT) for b in acc_bars],
                lag_ratio=0.15,
            ),
            run_time=0.8,
        )

        # ── Cost ──
        cost_title = self.txt("Cost (k tokens)", font_size=20,
                              color=MUTED, weight=BOLD)
        cost_title.set_y(-0.7)
        cost_title.align_to(np.array([LABEL_LEFT, 0, 0]), LEFT)
        self.play(FadeIn(cost_title), run_time=0.2)

        max_cost = 307.6
        cost_items = []
        for i, (name, acc, cost, col) in enumerate(methods):
            y = -1.3 - i * (BAR_H + ROW_GAP)
            cost_items.append(make_bar_row(name, cost, max_cost, "k", col, y))

        cost_bars = VGroup(*[item[1] for item in cost_items])
        cost_extras = VGroup(*[VGroup(item[0], item[2]) for item in cost_items])

        self.play(
            *[FadeIn(e) for e in cost_extras],
            LaggedStart(
                *[GrowFromEdge(b, LEFT) for b in cost_bars],
                lag_ratio=0.15,
            ),
            run_time=0.8,
        )

        # Cost saving annotation
        saving = self.txt("~50% less cost", font_size=20,
                          color=GREEN, weight=BOLD)
        saving.next_to(cost_bars[-1], RIGHT, buff=1.0)

        arrow = Arrow(
            saving.get_left(), cost_bars[-1].get_right() + RIGHT * 0.15,
            buff=0.08, color=GREEN, stroke_width=2,
            max_tip_length_to_length_ratio=0.3,
        )

        self.play(
            FadeIn(saving, shift=LEFT * 0.2),
            GrowArrow(arrow),
            run_time=0.5,
        )

        # Punchline
        punchline = self.txt(
            "Think deep, not just long.",
            font_size=36, color=YELLOW, weight=BOLD,
        )
        punchline.move_to(DOWN * 3.2)
        self.play(FadeIn(punchline, scale=0.85), run_time=0.6)
        self.wait(1.0)

    # ── Scene 6: Closing ─────────────────────────────────────────────

    def scene_6_closing(self):
        """Credits and links."""
        # Stacked title lines to avoid Pango wrapping
        t1 = self.txt("Think Deep,", font_size=46, weight=BOLD)
        t2 = self.txt("Not Just Long", font_size=46, weight=BOLD)
        title = VGroup(t1, t2).arrange(DOWN, buff=0.15, center=True)

        sub = self.txt("University of Virginia + Google",
                       font_size=24, color=MUTED)

        tg = VGroup(title, sub).arrange(DOWN, buff=0.35, center=True)
        tg.move_to(UP * 1.0)

        link = self.txt("arxiv.org/abs/2602.13517",
                        font_size=22, color=ACCENT)
        link.move_to(DOWN * 0.8)

        # Short author lines to avoid wrapping
        a1 = self.txt("Wei-Lin Chen, Liqian Peng, Tian Tan,",
                       font_size=15, color=MUTED)
        a2 = self.txt("Chao Zhao, Blake J. Chen, Ziqian Lin,",
                       font_size=15, color=MUTED)
        a3 = self.txt("Alec Go, Yu Meng",
                       font_size=15, color=MUTED)
        authors = VGroup(a1, a2, a3).arrange(DOWN, buff=0.06, center=True)
        authors.move_to(DOWN * 2.0)

        self.play(
            FadeIn(tg, shift=DOWN * 0.2),
            FadeIn(link, shift=UP * 0.15),
            FadeIn(authors, shift=UP * 0.1),
            run_time=0.7,
        )
        self.wait(1.5)
