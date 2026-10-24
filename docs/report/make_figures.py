"""Charts for the report. Every number below is copied from the project's result tables
(results/table_*.md, Colab runs of 20-24 Oct 2026)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import glob as _g, os as _o
for _f in _g.glob(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "fonts", "OpenSans-*.ttf")): _fm.fontManager.addfont(_f)

from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Open Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Open Sans", "mathtext.it": "Open Sans:italic", "mathtext.bf": "Open Sans:bold", "mathtext.fallback": "stixsans", "axes.unicode_minus": True,
    "axes.formatter.use_mathtext": True, "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#898781", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#C8102E", "#1A1A1A", "#E8737F", "#A6A6A6", "#E8737F", "#444444", "#555555", "#C8102E")
INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
FAM = {"Part 1 (classic)": BLUE, "Encoder comparison": ORANGE, "Part 2 (attention)": AQUA,
       "Pretrained reference": YELLOW, "Human": VIOLET}
W1, W2 = 6.6, 3.4  # two-column layout: full width ~6.7 in, column ~3.27 in


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Table A (beam 5, 3 seeds)
A = {  # config: (BLEU-4, sd, CIDEr, sd, METEOR, sd)
    "LSTM": {"frozen": (0.168, 0.006, 0.377, 0.024, 0.175, 0.004), "fine-tuned": (0.172, 0.004, 0.392, 0.016, 0.179, 0.002)},
    "GRU": {"frozen": (0.173, 0.001, 0.395, 0.006, 0.179, 0.002), "fine-tuned": (0.179, 0.002, 0.414, 0.004, 0.183, 0.001)},
    "Hybrid LSTM$\\rightarrow$GRU": {"frozen": (0.174, 0.004, 0.401, 0.006, 0.179, 0.000), "fine-tuned": (0.184, 0.002, 0.423, 0.003, 0.183, 0.001)},
}


def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.15))
    decs = list(A)
    x = np.arange(len(decs))
    w = 0.36
    for ax, (k, lab, lo) in zip(axes, [(2, "CIDEr", 0.30), (0, "BLEU-4", 0.14)]):
        for j, (enc, col) in enumerate([("frozen", "#F4A6AE"), ("fine-tuned", BLUE)]):
            m = [A[d][enc][k] for d in decs]
            s = [A[d][enc][k + 1] for d in decs]
            b = ax.bar(x + (j - 0.5) * w, m, w * 0.92, color=col, label=f"ResNet50 {enc}", zorder=2)
            ax.errorbar(x + (j - 0.5) * w, m, yerr=s, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
            for xi, mi, si in zip(x + (j - 0.5) * w, m, s):
                ax.text(xi, mi + si + (0.002 if k == 2 else 0.0008), f"{mi:.3f}", ha="center", va="bottom", fontsize=7.3, color=INK2)
        ax.set_xticks(x, ["LSTM", "GRU", "Hybrid"])
        ax.set_ylim(lo, max(A[d]["fine-tuned"][k] for d in decs) * 1.07)
        ax.set_ylabel(lab)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2)
    save(fig, "fig_ablation")


# ---------------------------------------------------------------- all models CIDEr (Table D + F)
ALL = [  # label, family, CIDEr, sd
    ("C1 LSTM", "Part 1 (classic)", 0.377, 0.024), ("C2 GRU", "Part 1 (classic)", 0.395, 0.006),
    ("C3 Hybrid", "Part 1 (classic)", 0.401, 0.006), ("C4 LSTM ft", "Part 1 (classic)", 0.392, 0.016),
    ("C5 GRU ft", "Part 1 (classic)", 0.414, 0.004), ("C6 Hybrid ft", "Part 1 (classic)", 0.423, 0.003),
    ("Hybrid + VGG16", "Encoder comparison", 0.380, 0.011), ("Hybrid + CLIP", "Encoder comparison", 0.582, 0.014),
    ("A1 LSTM + attention", "Part 2 (attention)", 0.524, 0.008), ("A2 Transformer", "Part 2 (attention)", 0.500, 0.007),
    ("BLIP-base", "Pretrained reference", 0.665, None), ("BLIP-large", "Pretrained reference", 0.693, None),
]


def fig_all_models():
    d = sorted(ALL, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    y = np.arange(len(d))
    ax.barh(y, [r[2] for r in d], 0.66, color=[FAM[r[1]] for r in d], zorder=2)
    for yi, r in zip(y, d):
        if r[3]:
            ax.errorbar(r[2], yi, xerr=r[3], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
        ax.text(r[2] + (r[3] or 0) + 0.008, yi, f"{r[2]:.3f}", va="center", fontsize=8.3, color=INK2)
    ax.set_yticks(y, [r[0] for r in d])
    ax.set_xlim(0, 0.76)
    ax.set_xlabel("CIDEr on the 1000 test images (mean $\\pm$ s.d. over 3 seeds; BLIP: single run)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAM.items() if f != "Human"],
              frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_all_models")


# ---------------------------------------------------------------- encoder comparison (Table F)
def fig_encoders():
    enc = [("VGG16\n(2014)", 0.380, 0.011, 0.167, 0.009), ("ResNet50\n(2015)", 0.401, 0.006, 0.174, 0.004),
           ("CLIP ViT-B/32\n(2021)", 0.582, 0.014, 0.233, 0.006)]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.9), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    x = np.arange(3)
    cols = ["#F4A6AE", BLUE, ORANGE]
    ax.bar(x, [e[1] for e in enc], 0.6, color=cols, zorder=2)
    ax.errorbar(x, [e[1] for e in enc], yerr=[e[2] for e in enc], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    for xi, e in zip(x, enc):
        ax.text(xi, e[1] + e[2] + 0.012, f"{e[1]:.3f}", ha="center", fontsize=8.3, color=INK2)
    ax.text(2, 0.70, "+45% vs ResNet50", ha="center", fontsize=8.8, color="#0b0b0b", fontweight="bold")
    ax.set_xticks(x, [e[0] for e in enc])
    ax.set_ylabel("CIDEr")
    ax.set_ylim(0, 0.78)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(a) Changing the encoder", loc="left")

    ax = axes[1]
    eff = [("Decoder type", 0.024), ("Fine-tuning", 0.022),
           ("Attention", 0.100), ("Encoder (CLIP)", 0.180)]
    y = np.arange(len(eff))
    ax.barh(y, [e[1] for e in eff], 0.6, color=[MUTED, MUTED, AQUA, ORANGE], zorder=2)
    for yi, e in zip(y, eff):
        ax.text(e[1] + 0.004, yi, f"+{e[1]:.3f}", va="center", fontsize=8.3, color=INK2)
    ax.set_yticks(y, [e[0] for e in eff])
    ax.set_xlim(0, 0.225)
    ax.set_xlabel("CIDEr gained")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(b) Size of each design choice", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_encoders")


# ---------------------------------------------------------------- accuracy illusion (Table B)
def fig_illusion():
    groups = ["Paper 1, hybrid\n(reported)", "Paper-protocol\nreplication", "C6 hybrid,\nfine-tuned (ours)"]
    padded = [0.8932, 0.788, 0.149]
    honest = [None, 0.411, 0.413]
    fig, ax = plt.subplots(figsize=(W2 + 0.4, 2.2))
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, padded, w * 0.92, color=ORANGE, label="Padding counted", zorder=2)
    ax.bar(x[1:] + w / 2, honest[1:], w * 0.92, color=BLUE, label="Padding ignored", zorder=2)
    for xi, v in zip(x - w / 2, padded):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.3, color=INK2)
    for xi, v in zip(x[1:] + w / 2, honest[1:]):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.3, color=INK2)
    ax.text(x[0] + w / 2, 0.03, "not\nreported", ha="center", fontsize=7.8, color=MUTED)
    ax.set_xticks(x, groups, fontsize=8.3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Token accuracy (test set)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_illusion")


# ---------------------------------------------------------------- greedy vs beam (Table C)
def fig_decoding():
    decs = ["LSTM", "GRU", "Hybrid"]
    p_g, p_b = [0.2698, 0.3986, 0.5142], [0.4286, 0.4935, 0.6034]
    o_g, o_b = [0.525, 0.512, 0.528], [0.569, 0.568, 0.572]
    o_gs, o_bs = [0.019, 0.032, 0.030], [0.003, 0.012, 0.022]
    fig, ax = plt.subplots(figsize=(W2 + 0.4, 2.2))
    x = np.arange(3)
    w = 0.2
    for off, vals, sd, col, lab in [(-1.5, p_g, None, "#BDBDBD", "Paper 1 greedy*"),
                                    (-0.5, p_b, None, ORANGE, "Paper 1 beam 3*"),
                                    (0.5, o_g, o_gs, "#F4A6AE", "Ours greedy"),
                                    (1.5, o_b, o_bs, BLUE, "Ours beam 3")]:
        ax.bar(x + off * w, vals, w * 0.9, color=col, label=lab, zorder=2)
        if sd:
            ax.errorbar(x + off * w, vals, yerr=sd, fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5)
    ax.set_xticks(x, decs)
    ax.set_ylabel("BLEU-1")
    ax.set_ylim(0, 0.68)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.1, columnspacing=0.8, handlelength=1.2)
    fig.tight_layout()
    save(fig, "fig_decoding")


# ---------------------------------------------------------------- significance (Table H, CIDEr)
H = [  # comparison, diff, lo, hi, p_holm
    ("GRU vs LSTM", 0.018, 0.003, 0.033, 0.049),
    ("Hybrid vs LSTM", 0.024, 0.009, 0.039, 0.006),
    ("Hybrid vs GRU", 0.006, -0.009, 0.021, 0.400),
    ("Fine-tuning: LSTM", 0.015, 0.005, 0.026, 0.012),
    ("Fine-tuning: GRU", 0.019, 0.008, 0.031, 0.011),
    ("Fine-tuning: hybrid", 0.022, 0.009, 0.035, 0.008),
    ("ResNet50 vs VGG16", 0.021, 0.003, 0.039, 0.049),
    ("CLIP vs ResNet50", 0.180, 0.155, 0.206, 0.001),
    ("Attention LSTM vs C6", 0.100, 0.078, 0.123, 0.001),
    ("Transformer vs attention LSTM", -0.023, -0.040, -0.006, 0.028),
    ("CLIP encoder vs attention LSTM", 0.058, 0.032, 0.085, 0.001),
]


def fig_forest():
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    y = np.arange(len(H))[::-1]
    for yi, (lab, d, lo, hi, p) in zip(y, H):
        col = BLUE if p < 0.05 else MUTED
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(d, yi, "o", color=col, ms=4.5, zorder=3, mec="white", mew=0.6)
        ptxt = f"$p={p:.3f}$"
        ax.text(0.222, yi, ptxt, va="center", fontsize=8.1, color=INK2 if p < 0.05 else MUTED)
    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y, [h[0] for h in H])
    ax.set_xlim(-0.05, 0.245)
    ax.set_xlabel("Difference in CIDEr (A$-$B), 95% bootstrap interval")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", lw=1.6, label="Real difference ($p<0.05$)"),
                       plt.Line2D([], [], color=MUTED, marker="o", lw=1.6, label="Within noise")],
              frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.45, 1.0))
    fig.tight_layout()
    save(fig, "fig_forest")


# ---------------------------------------------------------------- CLIPScore vs BLEU-4 (Table G)
G = [  # label, family, CLIP-S, BLEU-4
    ("C1", "Part 1 (classic)", 0.492, 0.168), ("C2", "Part 1 (classic)", 0.502, 0.172),
    ("C3", "Part 1 (classic)", 0.503, 0.174), ("C4", "Part 1 (classic)", 0.497, 0.172),
    ("C5", "Part 1 (classic)", 0.508, 0.179), ("C6", "Part 1 (classic)", 0.510, 0.184),
    ("VGG16", "Encoder comparison", 0.496, 0.167), ("CLIP enc.", "Encoder comparison", 0.594, 0.233),
    ("A1", "Part 2 (attention)", 0.549, 0.221), ("A2", "Part 2 (attention)", 0.551, 0.210),
    ("BLIP-base", "Pretrained reference", 0.612, 0.263), ("BLIP-large", "Pretrained reference", 0.641, 0.206),
]


def fig_clipscore():
    fig, ax = plt.subplots(figsize=(W2 + 0.4, 2.45))
    for lab, fam, cs, b4 in G:
        ax.scatter(b4, cs, s=26, color=FAM[fam], edgecolor="white", linewidth=0.6, zorder=3)
    for lab, fam, cs, b4 in G:
        if lab in ("BLIP-large", "BLIP-base", "CLIP enc.", "A1", "A2", "C1", "C6"):
            dx, dy, ha = {"BLIP-large": (-0.004, 0.0, "right"), "A2": (-0.004, 0.004, "right"),
                          "A1": (0.004, -0.008, "left"), "C1": (-0.003, -0.012, "right")}.get(lab, (0.004, -0.004, "left"))
            ax.text(b4 + dx, cs + dy, lab, fontsize=8.1, ha=ha, color="#0b0b0b")
    ax.axhline(0.712, color=VIOLET, lw=0.9, ls=(0, (3, 2)))
    ax.text(0.162, 0.703, "human captions 0.712", fontsize=8.1, color=VIOLET, va="top")
    ax.set_xlabel("BLEU-4 (agreement with reference wording)")
    ax.set_ylabel("CLIPScore (agreement with the photo)")
    ax.set_xlim(0.158, 0.272)
    ax.set_ylim(0.47, 0.73)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_clipscore")


# ---------------------------------------------------------------- hallucination (Table E)
E = [  # label, family, CHAIR-s, generic, repetition
    ("C1 LSTM", "Part 1 (classic)", 0.425, 0.257, 0.013), ("C2 GRU", "Part 1 (classic)", 0.366, 0.207, 0.044),
    ("C3 Hybrid", "Part 1 (classic)", 0.403, 0.226, 0.049), ("C6 Hybrid ft", "Part 1 (classic)", 0.381, 0.194, 0.044),
    ("A1 LSTM + attention", "Part 2 (attention)", 0.262, 0.059, 0.106), ("A2 Transformer", "Part 2 (attention)", 0.252, 0.053, 0.070),
    ("BLIP-base", "Pretrained reference", 0.113, 0.067, 0.018), ("BLIP-large", "Pretrained reference", 0.260, 0.049, 0.038),
]


def fig_hallucination():
    fig, axes = plt.subplots(1, 3, figsize=(W1, 2.9), sharey=True)
    y = np.arange(len(E))[::-1]
    for ax, k, title in zip(axes, [2, 3, 4], ["(a) Invented object", "(b) Generic caption", "(c) Repeated words"]):
        ax.barh(y, [e[k] for e in E], 0.64, color=[FAM[e[1]] for e in E], zorder=2)
        for yi, e in zip(y, E):
            ax.text(e[k] + 0.006, yi, f"{100 * e[k]:.0f}%", va="center", fontsize=7.8, color=INK2)
        ax.set_title(title, loc="left")
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0, max(e[k] for e in E) * 1.3)
    axes[0].set_yticks(y, [e[0] for e in E])
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_hallucination")


for f in [fig_ablation, fig_all_models, fig_encoders, fig_illusion, fig_decoding, fig_forest, fig_clipscore,
          fig_hallucination]:
    f()
print("charts written to", OUT)
