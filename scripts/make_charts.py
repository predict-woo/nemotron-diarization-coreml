"""Renders the README charts as static SVG, a light and a dark variant each (docs/img/).

Measured on an Apple M2 Pro, macOS 27, on battery (see the README for how each number was obtained).

    python scripts/make_charts.py
"""

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"

# --- data ---------------------------------------------------------------------------------------------------------

SPEED = [  # live streaming, Low latency, same three recordings (563.5 s of audio)
    ("PyTorch (original), CPU fp32", 6.04),
    ("CoreML, GPU fp32, one bucket", 19.2),
    ("CoreML, GPU fp32, five buckets", 24.9),
    ("CoreML, Neural Engine fp16", 42.4),
]
LENGTHS = [192, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576]
ANE_MS = [6.6, 8.3, 10.3, 11.3, 11.9, 12.7, 16.7, 17.7, 18.3, 21.7, 23.0, 23.7]
GPU_MS = [14.0, 17.6, 18.7, 21.4, 24.4, 24.5, 27.2, 28.5, 30.8, 32.8, 35.8, 34.9]
ANE_BUCKETS = [192, 288, 384, 480, 544]
DER = [  # (meeting, original %, Neural Engine %), DER vs ground truth
    ("2 speakers", 7.51, 7.62),
    ("3 speakers", 6.52, 6.58),
    ("4 speakers, 2 min", 11.59, 11.59),
    ("5 speakers", 5.65, 5.62),
    ("6 speakers", 10.40, 10.41),
    ("7 speakers", 10.67, 10.96),
    ("8 speakers", 7.95, 8.04),
    ("8 speakers (b)", 7.72, 7.52),
]
DER_OVERALL = (8.23, 8.27)

# --- theme (reference palette of the dataviz method) ---------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    border: str
    ink: str
    ink2: str
    grid: str
    axis: str
    muted: str  # de-emphasised marks
    band: str  # background band
    s1: str  # series 1: blue
    s2: str  # series 2: orange


LIGHT = Theme("light", "#ffffff", "#e3e3e3", "#191919", "#666666", "#ececec", "#c2c2c2", "#c9c9c9", "#f3f3f3",
              "#257de0", "#f47230")
DARK = Theme("dark", "#191919", "#2c2c2c", "#efefef", "#9c9c9c", "#2a2a2a", "#474747", "#4f4f4f", "#222222",
             "#338ced", "#e76624")
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'


class Svg:
    def __init__(self, width: int, height: int, theme: Theme, title: str):
        self.w, self.h, self.t = width, height, theme
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}" font-family=\'{FONT}\'>',
            f"<title>{escape(title)}</title>",
            f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" fill="{theme.surface}" '
            f'stroke="{theme.border}"/>',
        ]

    def add(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, size=12, fill=None, weight=400, anchor="start", baseline="auto"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill or self.t.ink}" text-anchor="{anchor}" dominant-baseline="{baseline}">'
                 f"{escape(s)}</text>")

    def line(self, x1, y1, x2, y2, stroke, width=1.0):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="{width}" shape-rendering="crispEdges"/>')

    def hbar(self, x0, x1, y, h, fill, r=4):
        """Horizontal bar from the baseline x0 to x1, square at the baseline, 4px rounded data end."""
        w = max(x1 - x0, 0.0)
        r = min(r, w, h / 2)
        self.add(f'<path d="M{x0:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
                 f'a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z" fill="{fill}"/>')

    def dot(self, x, y, fill, r=5):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 2}" fill="{self.t.surface}"/>'
                 f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>')

    def save(self, path: Path):
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n")


def header(svg: Svg, title: str, subtitle: str):
    svg.text(24, 34, title, size=16, weight=600)
    svg.text(24, 55, subtitle, size=12.5, fill=svg.t.ink2)


def legend(svg: Svg, x: float, y: float, items: list[tuple[str, str]]):
    for label, color in items:
        svg.add(f'<circle cx="{x + 5}" cy="{y - 4}" r="5" fill="{color}"/>')
        svg.text(x + 15, y, label, size=12, fill=svg.t.ink2)
        x += 15 + 6.6 * len(label) + 22


# --- charts --------------------------------------------------------------------------------------------------------

def speed_chart(t: Theme) -> Svg:
    W, H = 820, 318
    svg = Svg(W, H, t, "Live streaming speed: CoreML on the Neural Engine runs 7x faster than the original")
    header(svg, "Live streaming speed", "Low latency (1.04 s) · 563.5 s of audio · Apple M2 Pro on battery · "
                                         "higher is faster")
    left, right, top, row = 250, W - 215, 92, 44
    xmax = 45.0
    sx = lambda v: left + (right - left) * v / xmax
    bottom = top + row * len(SPEED)
    for v in range(0, 50, 10):
        svg.line(sx(v), top - 6, sx(v), bottom, t.grid)
        svg.text(sx(v), bottom + 18, f"{v}×", size=11.5, fill=t.ink2, anchor="middle")
    svg.text((left + right) / 2, bottom + 40, "× real time", size=12, fill=t.ink2, anchor="middle")
    # 4x requirement
    svg.line(sx(4), top - 14, sx(4), bottom, t.ink2)
    svg.text(sx(4) + 5, top - 16, "4× target", size=11.5, fill=t.ink2)
    base = SPEED[0][1]
    for i, (name, v) in enumerate(SPEED):
        y = top + i * row + (row - 20) / 2
        best = i == len(SPEED) - 1
        svg.text(left - 14, y + 10, name, size=13, fill=t.ink if best else t.ink2, anchor="end", baseline="middle",
                 weight=600 if best else 400)
        svg.hbar(sx(0), sx(v), y, 20, t.s1 if best else t.muted)
        label = f"{v:.1f}×" + ("" if i == 0 else f"  ({v / base:.1f}× the original)")
        svg.text(sx(v) + 8, y + 10, label, size=12.5, fill=t.ink if best else t.ink2, baseline="middle",
                 weight=600 if best else 400)
    svg.line(sx(0), top - 6, sx(0), bottom, t.axis)
    return svg


def latency_chart(t: Theme) -> Svg:
    W, H = 820, 390
    svg = Svg(W, H, t, "Encoder step latency versus step length on the GPU (fp32) and the Neural Engine (fp16)")
    header(svg, "One encoder step, by step length", "31-layer encoder at fixed lengths · lower is faster · "
                                                     "▲ Neural Engine buckets in use")
    legend(svg, 24, 82, [("GPU, fp32", t.s2), ("Neural Engine, fp16", t.s1)])
    left, right, top, bottom = 70, W - 190, 104, H - 74
    x0, x1, ymax = 160, 600, 40.0
    sx = lambda v: left + (right - left) * (v - x0) / (x1 - x0)
    sy = lambda v: bottom - (bottom - top) * v / ymax
    # steady-state band
    svg.add(f'<rect x="{sx(322):.1f}" y="{top:.1f}" width="{sx(541) - sx(322):.1f}" height="{bottom - top:.1f}" '
            f'fill="{t.band}"/>')
    svg.text((sx(322) + sx(541)) / 2, top + 14, "steady-state step lengths (322–541)", size=11.5, fill=t.ink2,
             anchor="middle")
    for v in range(0, 50, 10):
        svg.line(left, sy(v), right, sy(v), t.grid)
        svg.text(left - 10, sy(v), f"{v} ms" if v == 40 else f"{v}", size=11.5, fill=t.ink2, anchor="end",
                 baseline="middle")
    for v in range(192, 577, 64):
        svg.text(sx(v), bottom + 26, f"{v}", size=11.5, fill=t.ink2, anchor="middle")
    svg.text((left + right) / 2, bottom + 46, "step length (positions: speaker cache + FIFO + chunk)", size=12,
             fill=t.ink2, anchor="middle")
    svg.line(left, bottom, right, bottom, t.axis)
    for ms, color in ((GPU_MS, t.s2), (ANE_MS, t.s1)):
        pts = " ".join(f"{sx(l):.1f},{sy(v):.1f}" for l, v in zip(LENGTHS, ms))
        svg.add(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')
        for l, v in zip(LENGTHS, ms):
            svg.dot(sx(l), sy(v), color, r=4)
    for b in ANE_BUCKETS:
        x = sx(b)
        svg.add(f'<path d="M{x - 5:.1f},{bottom + 11:.1f} l5,-7 l5,7 z" fill="{t.s1}"/>')
    # direct end labels
    svg.text(sx(576) + 12, sy(GPU_MS[-1]), f"GPU fp32  {GPU_MS[-1]:.1f} ms", size=12.5, baseline="middle")
    svg.text(sx(576) + 12, sy(ANE_MS[-1]), f"Neural Engine  {ANE_MS[-1]:.1f} ms", size=12.5, baseline="middle",
             weight=600)
    return svg


def accuracy_chart(t: Theme) -> Svg:
    W = 820
    row = 30
    rows = DER + [("Overall", *DER_OVERALL)]
    top = 104
    H = top + row * len(rows) + 70
    svg = Svg(W, H, t, "Diarization error rate versus ground truth: the Neural Engine build matches the original")
    header(svg, "Diarization error rate vs. ground truth",
           "Synthetic meetings with known speaker turns · frame-level DER, no collar · lower is better · "
           "overlapping dots: equal")
    legend(svg, 24, 82, [("Original (PyTorch fp32)", t.s1), ("CoreML Neural Engine fp16", t.s2)])
    left, right = 190, W - 120
    x0, x1 = 4.0, 13.0
    sx = lambda v: left + (right - left) * (v - x0) / (x1 - x0)
    bottom = top + row * len(rows)
    for v in range(4, 14, 1):
        svg.line(sx(v), top - 4, sx(v), bottom, t.grid)
        svg.text(sx(v), bottom + 18, f"{v}%", size=11.5, fill=t.ink2, anchor="middle")
    svg.text((left + right) / 2, bottom + 40, "DER", size=12, fill=t.ink2, anchor="middle")
    for i, (name, ref, ane) in enumerate(rows):
        y = top + i * row + row / 2
        overall = name == "Overall"
        if overall:
            svg.line(left - 170, y - row / 2, right, y - row / 2, t.axis)
        svg.text(left - 14, y, name, size=12.5, fill=t.ink if overall else t.ink2, anchor="end", baseline="middle",
                 weight=600 if overall else 400)
        lo, hi = sorted((ref, ane))
        svg.line(sx(lo), y, sx(hi), y, t.axis, width=2)
        svg.dot(sx(ref), y, t.s1)
        svg.dot(sx(ane), y, t.s2)
        if overall:
            svg.text(sx(hi) + 12, y, f"{ref:.2f}% vs {ane:.2f}%", size=12.5, baseline="middle", weight=600)
    return svg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in (LIGHT, DARK):
        for name, chart in (("speed", speed_chart), ("latency", latency_chart), ("accuracy", accuracy_chart)):
            chart(theme).save(OUT / f"{name}-{theme.name}.svg")
    print(f"wrote {sorted(p.name for p in OUT.glob('*.svg'))}")


if __name__ == "__main__":
    main()
