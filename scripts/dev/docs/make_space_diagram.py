"""Write docs/_static/pipeline_details/space_tracking.svg, the coordinate-space diagram.

Three streams left to right (surface, anatomical, functional); each space is a box in its
stream's column, and each arrow is a pair of transform files (from-<A>_to-<B> and its
inverse). Edit STREAMS, SPACES or LINKS below and rerun:

    python scripts/dev/docs/make_space_diagram.py

The SVG is drawn at the page's text size (13 px names), so embed it at its natural width.
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[3] / "docs" / "_static" / "pipeline_details" / "space_tracking.svg"

# Layout (px)
BOX_W, BOX_H = 170, 50
COLUMN_X = {"surface": 110, "anat": 360, "func": 610}  # column centres
ROW_Y = {0: 70, 1: 175, 2: 290, 3: 410}  # row centres
HEADER_Y = 20
WIDTH, HEIGHT = 720, 480

STREAMS = {"surface": "Surface", "anat": "Anatomical (T1w / T2w)", "func": "Functional (BOLD)"}

# stream: (box fill, box border, header and entity text)
COLOURS = {
    "anat": ("#e8f6ee", "mediumseagreen", "#2e8b57"),
    "func": ("#f1ecfb", "mediumpurple", "#6a4fb3"),
    "surface": ("#f0f2f4", "#8a94a3", "#5f6b7a"),
}

# id: (stream, row, name, entity)
SPACES = {
    "t2w_scanner": ("anat", 0, "T2w scanner", "space-T2wScanner"),
    "t1w_scanner": ("anat", 1, "T1w scanner", "space-scanner"),
    "t1w": ("anat", 2, "T1w (conformed)", "space-T1w"),
    "template": ("anat", 3, "template", "space-<template>"),
    "bold_scanner": ("func", 1, "BOLD scanner", "input BOLD"),
    "bold": ("func", 2, "BOLD (conformed)", "space-bold"),
    "fastsurfer": ("surface", 2, "FastSurfer", "space-fsnative"),
}

# (from, to, label lines, dotted). Every link is drawn with arrows at both ends.
LINKS = [
    ("t2w_scanner", "t1w_scanner", ["T2wScanner ↔ scanner"], False),
    ("t1w_scanner", "t1w", ["scanner ↔ T1w"], False),
    ("t1w", "template", ["T1w ↔ <template>"], False),
    ("bold_scanner", "bold", ["scanner ↔ bold"], False),
    ("fastsurfer", "t1w", ["resample"], False),
    ("t1w", "bold", ["bold ↔ T1w"], False),
    ("bold", "template", ["bold ↔ <template>", "(no T1w for the session)"], True),
]

STYLE = """
      .box { stroke-width: 1.3; }
      .name { font-size: 13px; font-weight: 700; fill: #333; text-anchor: middle; }
      .entity { font-size: 11.5px; font-family: Menlo, Consolas, monospace; text-anchor: middle; }
      .head { font-size: 13px; font-weight: 700; text-anchor: middle; }
      .link { stroke: #777; stroke-width: 1.3; fill: none; }
      .dotted { stroke-dasharray: 4 3; }
      .label { font-size: 11px; fill: #555; }
"""


def stream_styles():
    return "".join(
        f"      .box.{s} {{ fill: {fill}; stroke: {border}; }}\n"
        f"      .head.{s}, .entity.{s} {{ fill: {ink}; }}\n"
        for s, (fill, border, ink) in COLOURS.items()
    )


def centre(space):
    stream, row, *_ = SPACES[space]
    return COLUMN_X[stream], ROW_Y[row]


def text(cls, x, y, s, anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    return f'  <text class="{cls}" x="{x:g}" y="{y:g}"{a}>{escape(s)}</text>'


def link(a, b, labels, dotted):
    (xa, ya), (xb, yb) = centre(a), centre(b)
    gap = 2  # keep arrow heads off the box border
    ends = 'marker-start="url(#arrow)" marker-end="url(#arrow)"'
    out = []
    if xa == xb:  # same stream: vertical, label to the right
        y1, y2 = sorted((ya, yb))
        out.append(f'  <line class="link" x1="{xa}" y1="{y1 + BOX_H / 2 + gap:g}" x2="{xa}" '
                   f'y2="{y2 - BOX_H / 2 - gap:g}" {ends}/>')
        for i, s in enumerate(labels):
            out.append(text("label", xa + 8, (y1 + y2) / 2 + 4 + 14 * i, s))
    elif ya == yb:  # same row: horizontal, label above
        x1, x2 = sorted((xa, xb))
        out.append(f'  <line class="link" x1="{x1 + BOX_W / 2 + gap:g}" y1="{ya}" '
                   f'x2="{x2 - BOX_W / 2 - gap:g}" y2="{ya}" {ends}/>')
        for i, s in enumerate(labels):
            out.append(text("label", (x1 + x2) / 2, ya - 8 + 14 * i, s, "middle"))
    else:  # across streams and rows: leave the upper box downward, enter the other from the side
        (xs, ys), (xt, yt) = ((xa, ya), (xb, yb)) if ya < yb else ((xb, yb), (xa, ya))
        side = 1 if xs > xt else -1
        sx, sy = xs, ys + BOX_H / 2 + gap
        tx, ty = xt + side * (BOX_W / 2 + gap), yt
        cls = "link dotted" if dotted else "link"
        out.append(f'  <path class="{cls}" d="M {sx:g} {sy:g} C {sx:g} {ty - 30:g}, '
                   f'{(sx + tx) / 2:g} {ty:g}, {tx:g} {ty:g}" {ends}/>')
        for i, s in enumerate(labels):  # under the curve's horizontal end
            out.append(text("label", (sx + tx) / 2 + 10, ty + 22 + 14 * i, s, "middle"))
    return out


def main():
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}"',
        "     font-family=\"Lato, 'Helvetica Neue', Arial, sans-serif\">",
        "  <!-- Generated by scripts/dev/docs/make_space_diagram.py; edit that, not this. -->",
        "  <defs>",
        '    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">',
        '      <path d="M0,0 L10,5 L0,10 z" fill="#777"/>',
        "    </marker>",
        f"    <style>{STYLE}{stream_styles()}    </style>",
        "  </defs>",
    ]
    lines += [text(f"head {s}", x, HEADER_Y, STREAMS[s]) for s, x in COLUMN_X.items()]
    for space, (stream, _, name, entity) in SPACES.items():
        x, y = centre(space)
        lines.append(f'  <rect class="box {stream}" x="{x - BOX_W / 2:g}" y="{y - BOX_H / 2:g}" '
                     f'width="{BOX_W}" height="{BOX_H}" rx="4"/>')
        lines.append(text("name", x, y - 4, name))
        lines.append(text(f"entity {stream}", x, y + 14, entity))
    for a, b, labels, dotted in LINKS:
        lines += link(a, b, labels, dotted)
    lines.append("</svg>")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
