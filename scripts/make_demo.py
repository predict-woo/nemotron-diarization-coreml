"""Records `nemotron-diarize live` in a pseudo-terminal and renders it as an animated SVG for the README.

The CLI plays a recording in real time (`live --file`), exactly as it shows the microphone. Its output is captured
with timestamps, replayed through a minimal terminal emulator and sampled into frames; the SVG flips through the
frames with a CSS animation (GitHub plays these in READMEs).

    python scripts/make_demo.py                       # docs/img/live-demo.svg
"""

from __future__ import annotations

import argparse
import fcntl
import html
import os
import pickle
import pty
import re
import select
import shutil
import struct
import tempfile
import termios
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# terminal palette (the window is dark in both GitHub themes)
BG, FG, DIM, TITLE = "#16181d", "#e3e5e8", "#7d828c", "#9aa0aa"
ANSI = {31: "#f0605a", 32: "#3fb950", 33: "#d8a63b", 34: "#4f8ff7", 35: "#c475e0", 36: "#3fc1d0"}


@dataclass(frozen=True)
class Style:
    fg: str | None = None
    bold: bool = False
    dim: bool = False


BLANK = (" ", Style())


class Terminal:
    """Just enough VT100 for the CLI: text, CR/LF, cursor up, erase line/below, SGR colours."""

    TOKEN = re.compile(r"\x1b\[([?0-9;]*)([A-Za-z])|\r|\n|[^\x1b\r\n]")
    PARTIAL = re.compile(r"\x1b(\[[?0-9;]*)?$")  # an escape sequence cut off at the end of a read

    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.grid = [[BLANK] * cols for _ in range(rows)]
        self.row = self.col = 0
        self.style = Style()
        self.pending = ""

    def feed(self, text: str) -> None:
        text = self.pending + text
        partial = self.PARTIAL.search(text)
        self.pending = partial.group(0) if partial else ""
        if partial:
            text = text[:partial.start()]
        for m in self.TOKEN.finditer(text):
            token = m.group(0)
            if m.group(2):
                self.csi(m.group(1), m.group(2))
            elif token == "\r":
                self.col = 0
            elif token == "\n":
                self.line_feed()
            elif token >= " ":
                if self.col >= self.cols:
                    self.col = 0
                    self.line_feed()
                self.grid[self.row][self.col] = (token, self.style)
                self.col += 1

    def line_feed(self) -> None:
        if self.row == self.rows - 1:
            self.grid.pop(0)
            self.grid.append([BLANK] * self.cols)
        else:
            self.row += 1

    def csi(self, params: str, final: str) -> None:
        if params.startswith("?"):
            return  # cursor visibility
        args = [int(p) if p else 0 for p in params.split(";")] if params else []
        if final == "A":
            self.row = max(self.row - max(args[0] if args else 1, 1), 0)
        elif final == "K":
            if (args or [0])[0] == 2:
                self.grid[self.row] = [BLANK] * self.cols
            else:
                self.grid[self.row][self.col:] = [BLANK] * (self.cols - self.col)
        elif final == "J":
            self.grid[self.row][self.col:] = [BLANK] * (self.cols - self.col)
            for r in range(self.row + 1, self.rows):
                self.grid[r] = [BLANK] * self.cols
        elif final == "m":
            self.sgr(args or [0])

    def sgr(self, args: list[int]) -> None:
        i = 0
        while i < len(args):
            a = args[i]
            if a == 0:
                self.style = Style()
            elif a == 1:
                self.style = Style(self.style.fg, True, self.style.dim)
            elif a == 2:
                self.style = Style(self.style.fg, self.style.bold, True)
            elif a in ANSI:
                self.style = Style(ANSI[a], self.style.bold, self.style.dim)
            elif a == 38 and i + 4 < len(args) and args[i + 1] == 2:
                r, g, b = args[i + 2:i + 5]
                self.style = Style(f"#{r:02x}{g:02x}{b:02x}", self.style.bold, self.style.dim)
                i += 4
            i += 1

    def snapshot(self) -> tuple[tuple[tuple[str, Style], ...], ...]:
        return tuple(tuple(row) for row in self.grid)


def record(cmd: list[str], cols: int, rows: int, env: dict[str, str]) -> list[tuple[float, bytes]]:
    """Runs `cmd` in a pseudo-terminal; returns its output as (seconds since start, bytes) chunks."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(cmd[0], cmd, env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    start, chunks = time.monotonic(), []
    while True:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if not ready:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append((time.monotonic() - start, data))
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n" + b"".join(c for _, c in chunks).decode(errors="replace"))
    return chunks


def frames(chunks, cols: int, rows: int, prompt: str, interval: float, hold: float):
    """Screen snapshots every `interval` seconds (identical neighbours merged) as (start, duration, screen)."""
    term = Terminal(cols, rows)
    term.feed(prompt + "\r\n")
    decoder = __import__("codecs").getincrementaldecoder("utf-8")()
    out: list[list] = []
    i, t, end = 0, 0.0, chunks[-1][0]
    while True:
        while i < len(chunks) and chunks[i][0] <= t:
            term.feed(decoder.decode(chunks[i][1]))
            i += 1
        shot = term.snapshot()
        if out and out[-1][2] == shot:
            out[-1][1] += interval
        else:
            out.append([t, interval, shot])
        if i >= len(chunks):
            break
        t += interval
    out[-1][1] = hold
    return out


def render_row(row, cw: float, lh: float, x0: float) -> str:
    """One terminal row as SVG at baseline y=0: text runs per style, timeline blocks (█) as rectangles."""
    parts = []
    c = 0
    while c < len(row):
        ch, style = row[c]
        run_end = c + 1
        while run_end < len(row) and row[run_end][1] == style and (row[run_end][0] == "█") == (ch == "█"):
            run_end += 1
        text = "".join(cell[0] for cell in row[c:run_end])
        colour = style.fg or FG
        if ch == "█":
            parts.append(f'<rect x="{x0 + c * cw:.1f}" y="{-lh * 0.72:.1f}" width="{(run_end - c) * cw:.1f}" '
                         f'height="{lh * 0.78:.1f}" rx="1.5" fill="{colour}"/>')
        elif text.strip():
            lead = len(text) - len(text.lstrip(" "))
            attrs = [f'x="{x0 + (c + lead) * cw:.1f}"']
            fill = DIM if style.dim and not style.fg else colour
            if fill != FG:
                attrs.append(f'fill="{fill}"')
            if style.dim and style.fg:
                attrs.append('opacity="0.6"')
            if style.bold:
                attrs.append('font-weight="700"')
            parts.append(f"<text {' '.join(attrs)}>{html.escape(text.strip(' '))}</text>")
        c = run_end
    return "".join(parts)


def build_svg(shots, cols: int, rows: int, title: str, still: int) -> str:
    font, cw, lh = 13.0, 7.83, 18.0
    pad, bar = 18.0, 34.0
    width = pad * 2 + cols * cw
    screen_h = rows * lh
    slot = screen_h + 4 * lh  # frames are stacked with a gap, so no neighbour shows at the clip edges
    height = bar + screen_h + pad
    total = sum(d for _, d, _ in shots)
    row_ids: dict = {}  # each distinct row is drawn once and placed with <use>
    defs, keyframes, groups = [], [], []
    for k, (start, _, shot) in enumerate(shots):
        keyframes.append(f"{100 * start / total:.3f}%{{transform:translateY({-k * slot:.0f}px)}}")
        uses = []
        for r, row in enumerate(shot):
            if all(ch == " " for ch, _ in row):
                continue
            if row not in row_ids:
                row_ids[row] = len(row_ids)
                defs.append(f'<g id="r{row_ids[row]}">{render_row(row, cw, lh, pad)}</g>')
            uses.append(f'<use href="#r{row_ids[row]}" y="{bar + lh * 0.95 + r * lh:.0f}"/>')
        groups.append(f'<g transform="translate(0 {k * slot:.0f})">{"".join(uses)}</g>')
    style = f"""
      .frames {{ animation: play {total:.2f}s steps(1, end) infinite; }}
      @keyframes play {{ {' '.join(keyframes)} 100%{{transform:translateY({-(len(shots) - 1) * slot:.0f}px)}} }}
      @media (prefers-reduced-motion: reduce) {{
        .frames {{ animation: none; transform: translateY({-still * slot:.0f}px); }}
      }}
      text {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
              font-size: {font}px; white-space: pre; }}
    """
    dots = "".join(f'<circle cx="{pad + 6 + 20 * i}" cy="{bar / 2 + 1}" r="6" fill="{c}"/>'
                   for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" xml:space="preserve" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="nemotron-diarize live: a scrolling timeline per speaker, with speaker turns logged as they end">
<style>{style}</style>
<defs><clipPath id="screen"><rect x="0" y="{bar}" width="{width:.0f}" height="{screen_h + pad / 2:.0f}"/></clipPath>
{chr(10).join(defs)}</defs>
<rect width="{width:.0f}" height="{height:.0f}" rx="10" fill="{BG}"/>
{dots}<text x="{width / 2:.0f}" y="{bar / 2 + 5:.0f}" text-anchor="middle" fill="{TITLE}" font-size="12">{html.escape(title)}</text>
<g clip-path="url(#screen)"><g class="frames" fill="{FG}">{''.join(groups)}</g></g>
</svg>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default=str(ROOT / "data/16k/synth_meeting_4spk_120s.wav"))
    ap.add_argument("--name", default="meeting.wav", help="file name shown in the recording")
    ap.add_argument("--bundle", default=str(ROOT / "dist/Nemotron3Diarization"))
    ap.add_argument("--cli", default=str(ROOT / ".build/release/nemotron-diarize"))
    ap.add_argument("--seconds", type=float, default=48, help="audio to play")
    ap.add_argument("--cols", type=int, default=96)
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--interval", type=float, default=0.36, help="seconds between frames")
    ap.add_argument("--hold", type=float, default=6.0, help="seconds the final frame stays")
    ap.add_argument("--out", default=str(ROOT / "docs/img/live-demo.svg"))
    ap.add_argument("--recording", help="reuse (or save to) this recording file")
    args = ap.parse_args()

    command = ["nemotron-diarize", "live", "--file", args.name]
    if args.recording and Path(args.recording).exists():
        chunks = pickle.loads(Path(args.recording).read_bytes())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / args.name
            shutil.copy(args.audio, audio)
            env = dict(os.environ, NEMOTRON_BUNDLE=args.bundle, TERM="xterm-256color")
            env.pop("NO_COLOR", None)
            chunks = record([args.cli, "live", "--file", str(audio), "--duration", str(args.seconds)],
                            args.cols, args.rows, env)
        if args.recording:
            Path(args.recording).write_bytes(pickle.dumps(chunks))
    shots = frames(chunks, args.cols, args.rows, "\x1b[2m$\x1b[0m " + " ".join(command), args.interval, args.hold)
    # still frame for reduced motion: the last one that shows the live view
    live = [k for k, (_, _, s) in enumerate(shots) if any("Ctrl-C" in "".join(c[0] for c in row) for row in s)]
    svg = build_svg(shots, args.cols, args.rows, " ".join(command), still=live[-1] if live else 0)
    Path(args.out).write_text(svg)
    print(f"{args.out}: {len(shots)} frames over {sum(d for _, d, _ in shots):.1f} s, {len(svg) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
