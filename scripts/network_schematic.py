import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_X = "#cfe8ff"      # x stream
C_Y = "#ffe0cc"      # y stream
C_C = "#e5d4ff"      # conditioning / modulation
C_BLK = "#f2f2f2"    # block bg
C_OUT = "#d6f5d6"    # outputs
C_OP = "#ffffff"     # primitive op
C_NORM = "#fff6cc"   # norm
EDGE = "#333333"


def box(x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=1.1, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, text="", color=EDGE, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.2, color=color, linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, text, ha="left",
                va="center", fontsize=7, style="italic", color="#444")


def shape(x, y, s):
    ax.text(x, y, s, ha="center", va="center", fontsize=7.5,
            family="monospace", color="#0057b7")


def stack(cx, top, items, w=3.0, h=0.62, gap=0.28):
    y = top
    for i, it in enumerate(items):
        text, fc = it[0], it[1]
        fs = it[2] if len(it) > 2 else 8
        box(cx - w / 2, y - h, w, h, text, fc, fs=fs)
        if i < len(items) - 1:
            arrow(cx, y - h, cx, y - h - gap)
        y = y - h - gap
    return y


def title(x, y, t):
    ax.text(x, y, t, ha="left", va="center", fontsize=10, fontweight="bold")


fig, ax = plt.subplots(figsize=(19, 15))
ax.set_xlim(0, 26); ax.set_ylim(0, 19); ax.axis("off")

ax.text(6, 18.5, "MMDiT  (canna/networks.py)", ha="center", fontsize=16, fontweight="bold")
ax.text(6, 18.05, "N=#sources · D=x_dim · H=hidden_dim · C=y_channels · (T,F) WDM · (n_t,n_f)=(T/16,F/16) patch grid · M=n_t·n_f",
        ha="center", fontsize=8.5, color="#555")
ax.plot([12.2, 12.2], [0.5, 17.6], color="#bbbbbb", lw=1.0)

# ======================================================================
# LEFT — global forward flow
# ======================================================================
box(0.6, 16.4, 2.6, 0.85, "x  source tokens", C_X, bold=True); shape(1.9, 16.15, "(N, D)")
box(4.7, 16.4, 2.6, 0.85, "y  WDM image", C_Y, bold=True); shape(6.0, 16.15, "(C, T, F)")
box(8.8, 16.4, 2.6, 0.85, "t  flow time", C_C, bold=True); shape(10.1, 16.15, "scalar")

box(0.6, 14.6, 2.6, 0.85, "x_embed (FeedForward)\n+ x_pos_embed", C_X, fs=8); shape(1.9, 14.35, "(N, H)")
arrow(1.9, 16.1, 1.9, 15.48)
box(4.7, 14.6, 2.6, 0.85, "Patchify  (4× Conv2d)", C_Y, fs=8); shape(6.0, 14.35, "(H, n_t, n_f)")
arrow(6.0, 16.1, 6.0, 15.48)
box(8.8, 14.6, 2.6, 0.85, "c_pos_embed (Sinusoidal)\n→ c_embed (FeedForward)", C_C, fs=7.5); shape(10.1, 14.35, "c: (H,)")
arrow(10.1, 16.1, 10.1, 15.48)

box(4.7, 13.1, 2.6, 0.8, "flatten (t f)→(tf) d\n+ y_pos_embed t/f", C_Y, fs=8); shape(6.0, 12.85, "(M, H)")
arrow(6.0, 14.55, 6.0, 13.94)

# ---- MMDiTBlock (expanded, parallel lanes) ----
box(0.4, 6.0, 9.2, 6.2, "", C_BLK)
ax.text(3.95, 11.85, "MMDiTBlock  ×  num_blocks", ha="center", fontsize=11, fontweight="bold")
XL, YL = 1.9, 6.0

arrow(10.1, 14.55, 8.7, 11.25, ls="--", color="#8850cc")
ax.plot([8.7, 8.7], [6.4, 11.25], ls="--", color="#8850cc", lw=1.1)
ax.text(8.85, 11.45, "4× Modulation(c)\n(shift, scale, gate)", ha="left", va="center", fontsize=7.5, color="#8850cc")

arrow(XL, 14.35, XL, 11.25)
arrow(YL, 13.05, YL, 11.25)

box(XL - 1.3, 10.55, 2.6, 0.65, "adaptive_norm (x)", C_NORM, fs=8)
box(YL - 1.3, 10.55, 2.6, 0.65, "adaptive_norm (y)", C_NORM, fs=8)

box(0.9, 9.4, 7.8, 0.85, "MultiStreamAttention   —  joint over concat(x ‖ y)  →  hx, hy", C_OP, fs=8.5)
arrow(XL, 10.55, XL, 10.25); arrow(YL, 10.55, YL, 10.25)

box(XL - 1.3, 8.45, 2.6, 0.6, "x += hx · gate0", C_X, fs=8)
box(YL - 1.3, 8.45, 2.6, 0.6, "y += hy · gate1", C_Y, fs=8)
arrow(XL, 9.4, XL, 9.05); arrow(YL, 9.4, YL, 9.05)

box(XL - 1.3, 7.2, 2.6, 0.9, "adaptive_norm (x)\n→ mlp_x  (H→2H→H)", C_X, fs=7.5)
box(YL - 1.3, 7.2, 2.6, 0.9, "adaptive_norm (y)\n→ mlp_y  (H→2H→H)", C_Y, fs=7.5)
arrow(XL, 8.45, XL, 8.1); arrow(YL, 8.45, YL, 8.1)

box(XL - 1.3, 6.2, 2.6, 0.55, "x += hx · gate2", C_X, fs=8)
box(YL - 1.3, 6.2, 2.6, 0.55, "y += hy · gate3", C_Y, fs=8)
arrow(XL, 7.2, XL, 6.75); arrow(YL, 7.2, YL, 6.75)

for yconn in (10.85, 8.7, 7.6, 6.45):
    arrow(8.7, yconn, 7.3, yconn, ls="--", color="#8850cc")

# ---- heads ----
box(0.6, 4.0, 3.6, 1.1, "x-head:\nadaptive_norm(c) → out_unembed", C_X, fs=8); shape(2.4, 3.8, "(N, 2D)")
arrow(XL, 6.2, XL, 5.15)
box(0.6, 2.0, 1.7, 0.9, "dx\nflow vel.", C_OUT, fs=8, bold=True); shape(1.45, 1.75, "(N, D)")
box(2.5, 2.0, 1.7, 0.9, "x_mle\npoint est.", C_OUT, fs=8, bold=True); shape(3.35, 1.75, "(N, D)")
arrow(1.9, 3.95, 1.45, 2.95); arrow(2.6, 3.95, 3.35, 2.95)
box(5.2, 4.0, 4.2, 1.1, "y-head:\nadaptive_norm(c) → rearrange → Unpatchify (resize-conv)", C_Y, fs=7.5)
arrow(YL, 6.2, 6.5, 5.15)
box(6.4, 2.0, 2.0, 0.9, "y_recon\nWDM reconstr.", C_OUT, fs=8, bold=True); shape(7.4, 1.75, "(C, T', F')")
arrow(7.3, 3.95, 7.4, 2.95)
box(9.9, 2.0, 1.9, 3.0, "push()\ninference:\nRK4 ODE\nt: 0→1\nuses dx +\nexp_map\n→ samples\n(N, D)", C_NORM, fs=8)

# ======================================================================
# RIGHT — component internals
# ======================================================================
ax.text(12.6, 17.9, "Component internals", fontsize=13, fontweight="bold")
ax.text(12.6, 17.5, "white = primitive op · yellow = norm · SiLU throughout (ACTIVATION = jax.nn.silu)",
        fontsize=8, color="#555")

# FeedForward
title(12.6, 16.9, "FeedForward(in, out, width)")
stack(14.6, 16.55, [
    ("Linear(in → width)", C_OP),
    ("SiLU", C_NORM),
    ("Linear(width → out)", C_OP),
], w=3.4)

# SinusoidalEmbed
title(18.8, 16.9, "SinusoidalEmbed(dim, period=2π)")
stack(20.9, 16.55, [
    ("[ sin(f·t) ‖ cos(f·t) ]   (2·dim)", C_OP, 7.5),
    ("FeedForward(2dim → dim → dim)", C_C, 7.5),
], w=4.2)
ax.text(18.65, 14.1, "geometric freqs f = period^−linspace(0,1,dim);  t scalar → (dim)", fontsize=7.5, color="#444")

# Modulation
title(12.6, 13.5, "Modulation(dim)")
stack(14.6, 13.15, [
    ("SiLU(c)", C_NORM),
    ("Linear(dim → 3·dim)  [zero-init]", C_OP, 7.5),
    ("split → shift, scale, gate", C_C, 7.5),
], w=3.6)
ax.text(12.6, 10.65, "zero-init ⇒ identity modulation at start", fontsize=7.5, color="#444")

# adaptive_norm
title(18.8, 13.5, "adaptive_norm(x, scale, shift)")
stack(20.9, 13.15, [
    ("(x − mean) / (std + eps)", C_NORM),
    ("· (1 + scale) + shift", C_C),
], w=3.8)

# Patchify / Unpatchify  (fixed 4-stage strided conv down/up-sampling)
ax.text(12.55, 9.72, "Patchify(C, H)  ↓16", fontsize=9, fontweight="bold")
botL = stack(13.95, 9.35, [
    ("Conv2d(C → H/8)  k3 s2 p1", C_OP, 6.4),
    ("GroupNorm → SiLU", C_NORM, 6.4),
    ("Conv2d(H/8 → H/4)", C_OP, 6.4),
    ("GroupNorm → SiLU", C_NORM, 6.4),
    ("Conv2d(H/4 → H/2)", C_OP, 6.4),
    ("GroupNorm → SiLU", C_NORM, 6.4),
    ("Conv2d(H/2 → H)  [bare]", C_OP, 6.4),
], w=2.8, h=0.48, gap=0.13)

ax.text(15.7, 9.72, "Unpatchify(C, H)  ↑16", fontsize=9, fontweight="bold")
stack(17.1, 9.35, [
    ("↑2 nearest → Conv2d(H → H/2) k3 s1 p1", C_OP, 5.4),
    ("GroupNorm → SiLU", C_NORM, 6.4),
    ("↑2 nearest → Conv2d(H/2 → H/4)", C_OP, 5.7),
    ("GroupNorm → SiLU", C_NORM, 6.4),
    ("↑2 nearest → Conv2d(H/4 → H/8)", C_OP, 5.7),
    ("GroupNorm → SiLU", C_NORM, 6.4),
    ("↑2 nearest → Conv2d(H/8 → C)  [bare]", C_OP, 5.5),
], w=2.8, h=0.48, gap=0.13)
ax.text(12.55, botL - 0.28, "down: 4 stride-2 convs (↓16 t,f).  up: 4× (nearest ↑2 + stride-1 conv) resize-conv (↑16 t,f);", fontsize=6.8, color="#444")
ax.text(12.55, botL - 0.55, "GroupNorm groups = channels; final (un)patch conv is bare (no norm/act).", fontsize=6.8, color="#444")

# MultiStreamAttention
title(18.8, 9.7, "MultiStreamAttention(dim, heads)")
stack(21.1, 9.35, [
    ("qkv_proj_x(x)   qkv_proj_y(y)\n Linear(dim → 3·dim)", C_OP, 7),
    ("concat streams → (N+M, 3·dim)", C_OP, 7),
    ("split heads → (N+M, H, ·)", C_OP, 7),
    ("q,k = rms_norm(q), rms_norm(k)", C_NORM, 7),
    ("dot_product_attention(q,k,v)", C_OP, 7),
    ("split streams back", C_OP, 7),
    ("out_proj_x(x)   out_proj_y(y)", C_OP, 7),
], w=4.4, h=0.5, gap=0.16)
ax.text(18.8, 4.5, "qkv / out projections: use_bias=False (default)", fontsize=7, color="#444")

# legend
box(12.6, 3.1, 3.4, 0.5, "primitive op (Linear/Conv/attn)", C_OP, fs=7.5)
box(12.6, 2.5, 3.4, 0.5, "normalization", C_NORM, fs=8)
box(16.3, 3.1, 2.6, 0.5, "x-stream", C_X, fs=8)
box(16.3, 2.5, 2.6, 0.5, "y-stream", C_Y, fs=8)
box(19.2, 3.1, 3.2, 0.5, "conditioning / modulation", C_C, fs=8)
box(19.2, 2.5, 3.2, 0.5, "outputs", C_OUT, fs=8)

fig.savefig("/home/dsartor_umass_edu/CANNA/outputs/network_schematic.pdf", bbox_inches="tight")
print("saved")
