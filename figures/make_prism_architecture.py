#!/usr/bin/env python3
"""PRISM architecture figure (format-agnostic controller) as SVG; render to PDF/PNG with cairosvg.

python figures/make_prism_architecture.py            # writes figures/prism_architecture.svg
"""
from pathlib import Path

W, H = 1800, 790
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
INK, MUTED, RULE = "#1f2433", "#5b6275", "#c9cedb"
RECENT, RECENT_FILL = "#2f6fd6", "#dce8fb"
HIST, HIST_FILL = "#9aa1b2", "#eceef3"
BAND, BAND_FILL = "#d98a1c", "#fbecd3"
OK, OK_FILL = "#2f9a5c", "#dcf2e4"
NO, NO_FILL = "#c8473f", "#f9dedc"
VLM, VLM_FILL = "#6b4fbf", "#e9e3f8"
CLIP, CLIP_FILL = "#16897f", "#d7f1ee"
out = []


def add(s):
    out.append(s)


def text(x, y, s, size=15, weight="normal", color=INK, anchor="start", style="normal"):
    s = str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" fill="{color}" text-anchor="{anchor}">{s}</text>')


def rect(x, y, w, h, fill, stroke, r=10, sw=1.6, dash=None, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw}"{d} opacity="{opacity}"/>')


def arrow(x1, y1, x2, y2, color=INK, sw=2.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} '
        f'marker-end="url(#head)"/>')


def frame(x, y, w, h, kind, seed, stroke):
    """A small abstract video frame: sky, ground, and one object whose shape/colour varies with seed."""
    sky = {"hist": "#d9dde6", "band": "#f6dcb0", "recent": "#c9dbf7"}[kind]
    ground = {"hist": "#b9bfcc", "band": "#e8b46a", "recent": "#8fb3ea"}[kind]
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{sky}" stroke="{stroke}" stroke-width="1.6"/>')
    add(f'<rect x="{x + 1}" y="{y + h * .62}" width="{w - 2}" height="{h * .38 - 1}" rx="3" fill="{ground}"/>')
    cx = x + w * (.3 + .4 * ((seed * 37) % 10) / 10)
    obj = ["#e0553f", "#3f7fe0", "#2f9a5c", "#8a5bd6"][seed % 4]
    if seed % 2:
        add(f'<circle cx="{cx}" cy="{y + h * .5}" r="{h * .17}" fill="{obj}"/>')
    else:
        add(f'<rect x="{cx - h * .16}" y="{y + h * .34}" width="{h * .32}" height="{h * .3}" rx="2" fill="{obj}"/>')


def badge(x, y, ok):
    color = OK if ok else NO
    add(f'<circle cx="{x}" cy="{y}" r="12" fill="{color}"/>')
    text(x, y + 5, "✓" if ok else "✗", 15, "bold", "#ffffff", "middle")


def snowflake(x, y, size=13):
    text(x, y, "❄", size, "normal", "#4a90c8", "middle")


def block(x, y, w, h, color, fill, number, title):
    rect(x, y, w, h, "#ffffff", color, 12, 1.8)
    add(f'<path d="M{x + 12},{y} h{w - 24} a12,12 0 0 1 12,12 v26 h-{w} v-26 a12,12 0 0 1 12,-12 z" fill="{fill}" '
        f'stroke="{color}" stroke-width="1.8"/>')
    add(f'<circle cx="{x + 22}" cy="{y + 19}" r="11" fill="{color}"/>')
    text(x + 22, y + 24, str(number), 13, "bold", "#ffffff", "middle")
    text(x + 40, y + 25, title, 15.5, "bold", INK)


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs><marker id="head" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
    '<marker id="headred" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{NO}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ---------------------------------------------------------------- streaming input
rect(20, 14, W - 40, 205, "#fafbfd", RULE, 14, 1.4)
text(40, 44, "Streaming input (1 fps)", 17, "bold")
text(275, 44, "the frozen VLM only ever sees the exact Recent-6 window plus at most 3 admitted history frames",
     14, "normal", MUTED, style="italic")
n, fw, fh, gap, x0, y0 = 22, 58, 42, 9, 52, 92
kinds = ["hist"] * 8 + ["band"] * 8 + ["recent"] * 6
for i, kind in enumerate(kinds):
    stroke = {"hist": HIST, "band": BAND, "recent": RECENT}[kind]
    frame(x0 + i * (fw + gap), y0, fw, fh, kind, i, stroke)
xs = lambda i: x0 + i * (fw + gap)
for a, b, color, label in ((0, 7, HIST, "older history: out of band, never scored"),
                           (8, 15, BAND, "admissible band  3 s ≤ d ≤ 30 s  →  candidates"),
                           (16, 21, RECENT, "exact Recent-6 context  R₆ = K₀")):
    l, r = xs(a), xs(b) + fw
    add(f'<path d="M{l},{y0 - 8} v-8 h{r - l} v8" fill="none" stroke="{color}" stroke-width="2"/>')
    text((l + r) / 2, y0 - 22, label, 14, "bold", color, "middle")
ty = y0 + fh + 22
add(f'<line x1="{x0}" y1="{ty}" x2="{xs(21) + fw + 18}" y2="{ty}" stroke="{INK}" stroke-width="1.6" marker-end="url(#head)"/>')
for i, label in ((0, "t = 0"), (8, "t_q − 30 s"), (16, "t_q − 6 s"), (21, "t_q")):
    xx = xs(i) + (fw if i == 21 else 0)
    add(f'<line x1="{xx}" y1="{ty - 5}" x2="{xx}" y2="{ty + 5}" stroke="{INK}" stroke-width="1.6"/>')
    text(xx, ty + 22, label, 13.5, "normal", MUTED, "middle")
# question bubble
bx, by = 1560, 70
rect(bx, by, 220, 118, "#ffffff", INK, 12, 1.6)
add(f'<path d="M{bx + 20},{by + 118} l-14,18 l34,-18 z" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
add(f'<rect x="{bx + 21}" y="{by + 114}" width="18" height="6" fill="#ffffff"/>')
text(bx + 14, by + 26, "Question q at t_q", 14.5, "bold")
text(bx + 14, by + 50, "“What did the person put", 13.5)
text(bx + 14, by + 68, " on the table?”", 13.5)
text(bx + 14, by + 92, "open-ended or multiple-choice:", 12.5, color=MUTED, style="italic")
text(bx + 14, by + 108, "one controller, no option scores", 12.5, color=MUTED, style="italic")

# ---------------------------------------------------------------- pipeline blocks
Y, BH = 262, 360
B = [(28, 236), (290, 262), (578, 300), (904, 214), (1144, 404), (1574, 198)]

# (1) temporal pre-filter
x, w = B[0]
block(x, Y, w, BH, BAND, BAND_FILL, 1, "Temporal pre-filter")
for k in range(8):
    kind = "band" if 2 <= k <= 6 else "hist"
    frame(x + 18 + (k % 4) * 52, Y + 62 + (k // 4) * 44, 46, 34, kind, k + 3, BAND if kind == "band" else HIST)
    if kind == "hist":
        add(f'<line x1="{x + 18 + (k % 4) * 52}" y1="{Y + 62 + (k // 4) * 44}" x2="{x + 64 + (k % 4) * 52}" '
            f'y2="{Y + 96 + (k // 4) * 44}" stroke="{NO}" stroke-width="2"/>')
text(x + 18, Y + 178, "keep h only if", 14)
text(x + 18, Y + 200, "d_min ≤ t_s(R₆) − t_e(h)", 14, style="italic")
text(x + 58, Y + 220, "≤ d_max", 14, style="italic")
text(x + 18, Y + 242, "(3 s, 30 s); horizon 64", 13.5, color=MUTED)
text(x + 18, Y + 270, "drops near-duplicates of R₆", 13.5, color=MUTED)
text(x + 18, Y + 290, "and stale evidence before", 13.5, color=MUTED)
text(x + 18, Y + 310, "any retrieval cost is paid", 13.5, color=MUTED)
text(x + 18, Y + 338, "→ candidate set H_t", 14.5, "bold", BAND)

# (2) CLIP-MMR
x, w = B[1]
block(x, Y, w, BH, CLIP, CLIP_FILL, 2, "CLIP-MMR retrieval")
for k, (score, kind) in enumerate(((.86, "band"), (.62, "band"), (.71, "band"), (.40, "band"))):
    yy = Y + 52 + k * 36
    frame(x + 18, yy, 44, 30, kind, k + 9, BAND)
    add(f'<rect x="{x + 72}" y="{yy + 9}" width="{150 * score}" height="13" rx="3" fill="{CLIP}" opacity=".75"/>')
    text(x + 78 + 150 * score, yy + 21, f"{score:.2f}", 12.5, color=MUTED)
snowflake(x + w - 20, Y + 25, 16)
text(x + 18, Y + 56 + 4 * 38 + 6, "query = question text only", 12.5, color=MUTED, style="italic")
text(x + 18, Y + 236, "r_i = cos(E_v(h_i), E_t(q))", 14, style="italic")
text(x + 18, Y + 262, "MMR: λ r_i − (1−λ) max_s sim(h_i, h_s)", 12.5, style="italic")
text(x + 18, Y + 286, "λ = 0.8 → diverse, relevant queue", 13.5, color=MUTED)
for k, label in enumerate(("h₁", "h₂", "h₃")):
    rect(x + 18 + k * 58, Y + 308, 48, 36, BAND_FILL, BAND, 8, 1.6)
    text(x + 42 + k * 58, Y + 332, label, 15, "bold", BAND, "middle")
text(x + 196, Y + 332, "ranked", 13.5, color=MUTED)

# (3) Recent sufficiency
x, w = B[2]
block(x, Y, w, BH, VLM, VLM_FILL, 3, "Recent sufficiency (K₀ = R₆)")
for k in range(6):
    frame(x + 16 + k * 45, Y + 56, 40, 30, "recent", k + 16, RECENT)
rect(x + 70, Y + 100, 160, 40, VLM_FILL, VLM, 10, 1.8)
text(x + 150, Y + 125, "frozen VLM", 14.5, "bold", VLM, "middle")
snowflake(x + 218, Y + 116, 14)
arrow(x + 150, Y + 88, x + 150, Y + 98, VLM, 1.8)
text(x + 16, Y + 166, "generates â₀ with token probabilities", 13.5, color=MUTED)
for k, (p1, p2) in enumerate(((.92, .04), (.71, .17), (.88, .06), (.55, .30))):
    bx2 = x + 22 + k * 66
    add(f'<rect x="{bx2}" y="{Y + 222 - 46 * p1}" width="18" height="{46 * p1}" fill="{VLM}" opacity=".85"/>')
    add(f'<rect x="{bx2 + 21}" y="{Y + 222 - 46 * p2}" width="18" height="{46 * p2}" fill="{VLM}" opacity=".35"/>')
add(f'<line x1="{x + 16}" y1="{Y + 223}" x2="{x + w - 16}" y2="{Y + 223}" stroke="{RULE}" stroke-width="1.4"/>')
text(x + 16, Y + 254, "S₀ = α M₀ + β E₀ + γ V₀", 15, "bold", INK)
text(x + 16, Y + 280, "M: top-2 token margin", 13.5, color=MUTED)
text(x + 16, Y + 300, "E: 1 − normalized token entropy", 13.5, color=MUTED)
text(x + 16, Y + 320, "V: CLIP support of (q ⊕ â₀)", 13.5, color=MUTED)
text(x + 16, Y + 342, "(α, β, γ) = (0.5, 0.2, 0.3)", 13.5, color=MUTED)

# (4) Admission gate
x, w = B[3]
block(x, Y, w, BH, INK, "#eef0f5", 4, "Admission gate")
cx, cy = x + w / 2, Y + 128
add(f'<path d="M{cx},{cy - 58} L{cx + 78},{cy} L{cx},{cy + 58} L{cx - 78},{cy} z" fill="#ffffff" stroke="{INK}" stroke-width="1.8"/>')
text(cx, cy - 6, "admit", 15, "bold", INK, "middle")
text(cx, cy + 14, "history?", 15, "bold", INK, "middle")
text(x + 16, Y + 222, "S₀ < τ_S  (recent unsure)", 13.5, style="italic")
text(x + 16, Y + 244, "  ∨", 14, "bold")
text(x + 16, Y + 266, "r₁ ≥ τ_R ∧ h₁ supports", 13.5, style="italic")
text(x + 16, Y + 286, "â₀ less than R₆ does", 13.5, style="italic")
text(x + 16, Y + 326, "no → answer â₀", 14, "bold", RECENT)
text(x + 16, Y + 346, "(identical to Recent-6)", 13, color=MUTED)

# (5) Progressive arbitration
x, w = B[4]
block(x, Y, w, BH, OK, OK_FILL, 5, "Progressive arbitration (B_mem = 3)")
rows = (("K₁", 1, True), ("K₂", 2, True), ("K₃", 3, False))
for r, (label, depth, ok) in enumerate(rows):
    yy = Y + 58 + r * 64
    text(x + 18, yy + 26, label, 16, "bold", INK)
    for k in range(depth):
        frame(x + 52 + k * 38, yy + 6, 34, 26, "band", k + 9, OK if (ok or k < depth - 1) else NO)
    for k in range(6):
        frame(x + 52 + depth * 38 + 8 + k * 28, yy + 6, 25, 26, "recent", k + 16, RECENT)
    badge(x + w - 30, yy + 16, ok)
    text(x + w - 30, yy + 44, "accept" if ok else "reject", 11.5, "bold", OK if ok else NO, "middle")
add(f'<path d="M{x + w - 44},{Y + 58 + 2 * 64 + 18} C{x + w - 70},{Y + 58 + 2 * 64 + 6} {x + w - 70},{Y + 58 + 64 + 40} '
    f'{x + w - 44},{Y + 58 + 64 + 22}" fill="none" stroke="{NO}" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#headred)"/>')
text(x + w - 18, Y + 268, "prefix rollback: K* = K₂", 13.5, "bold", NO, "end")
text(x + 18, Y + 306, "A_j = [M_j ≥ τ_M]·[S_(j−1) − S_j ≤ δ_S]·C_j·P_j", 14, style="italic")
text(x + 18, Y + 330, "C_j: a close frame must support the new answer", 13.5, color=MUTED)
text(x + 18, Y + 350, "P_j: answer changes, or S_j rises by ε", 13.5, color=MUTED)

# (6) Answer
x, w = B[5]
block(x, Y, w, BH, RECENT, RECENT_FILL, 6, "Answer")
for k in range(2):
    frame(x + 18 + k * 36, Y + 62, 32, 26, "band", k + 9, OK)
for k in range(6):
    frame(x + 18 + 2 * 36 + 4 + (k % 3) * 34, Y + 62 + (k // 3) * 30, 30, 26, "recent", k + 16, RECENT)
text(x + 18, Y + 152, "â = â_(j*) from K*", 15, "bold", INK)
text(x + 18, Y + 180, "deepest accepted state;", 13.5, color=MUTED)
text(x + 18, Y + 200, "no extra generation", 13.5, color=MUTED)
text(x + 18, Y + 236, "|K*| ≤ 6 + 3 frames", 14, style="italic")
text(x + 18, Y + 262, "j* ∈ {0, 1, 2, 3}", 14, style="italic")
text(x + 18, Y + 300, "no memory upkeep:", 13.5, "bold", INK)
text(x + 18, Y + 320, "only CLIP image", 13.5, color=MUTED)
text(x + 18, Y + 340, "embeddings are cached", 13.5, color=MUTED)

# arrows between blocks
mid = Y + BH / 2
for (xa, wa), (xb, _wb) in zip(B, B[1:]):
    arrow(xa + wa + 4, mid, xb - 6, mid)
# input to blocks
arrow(xs(12), ty + 30, B[0][0] + B[0][1] / 2, Y - 6, BAND, 1.8, "6 4")
arrow(xs(18) + 20, ty + 30, B[2][0] + B[2][1] / 2, Y - 6, RECENT, 1.8, "6 4")
arrow(bx + 30, by + 140, B[1][0] + B[1][1] - 30, Y - 6, INK, 1.8, "6 4")

# ---------------------------------------------------------------- legend
ly = 668
rect(20, ly - 16, W - 40, 112, "#fafbfd", RULE, 14, 1.4)
items = ((RECENT, RECENT_FILL, "exact Recent-6 frames (always kept, hash-identical)"),
         (BAND, BAND_FILL, "historical candidate frames (3–30 s band)"),
         (OK, OK_FILL, "accepted memory state"),
         (NO, NO_FILL, "rejected state → rollback to last accepted prefix"))
for k, (stroke, fill, label) in enumerate(items):
    lx = 44 + (k % 2) * 560
    yy = ly + 8 + (k // 2) * 34
    rect(lx, yy, 28, 20, fill, stroke, 5, 1.8)
    text(lx + 40, yy + 15, label, 14.5)
snowflake(1186, ly + 24, 18)
text(1204, ly + 24, "frozen, training-free: MiniCPM-V-4.6 + CLIP", 14.5)
text(1172, ly + 58, "MCQ and open-ended questions share one controller;", 13.5, color=MUTED, style="italic")
text(1172, ly + 78, "all signals come from the generated answer, never from options", 13.5, color=MUTED, style="italic")

add('</svg>')
target = Path(__file__).with_name('prism_architecture.svg')
target.write_text('\n'.join(out), encoding='utf-8')
print(target)
