"""Image figures from the notebook outputs (real Colab runs)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import glob as _g, os as _o
for _f in _g.glob(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "fonts", "OpenSans-*.ttf")): _fm.fontManager.addfont(_f)

from PIL import Image

S = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Open Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Open Sans", "mathtext.it": "Open Sans:italic", "mathtext.bf": "Open Sans:bold", "mathtext.fallback": "stixsans", "axes.unicode_minus": True, "font.size": 9})


def panels(path):
    """Cut a word strip into its square panels (image area only, no title text)."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    h = im.shape[0]
    band = im[int(h * 0.45):int(h * 0.75)]
    nonwhite = (np.abs(band - 255).sum(-1) > 30).mean(0) > 0.5
    runs, start = [], None
    for x, v in enumerate(nonwhite):
        if v and start is None:
            start = x
        if not v and start is not None:
            if x - start > 40:
                runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(nonwhite)))
    widths = [b - a for a, b in runs]
    med = np.median(widths)
    runs = [runs[0]] + [(a, b) for a, b in runs[1:] if abs((b - a) - med) < 0.15 * med]   # first = photo
    out = []
    for a, b in runs:
        col = im[:, a:b]
        rows = (np.abs(col - 255).sum(-1) > 30).mean(1) > 0.9   # rows that are (almost) all image
        best, cur, start = (0, 0), 0, 0
        for y, v in enumerate(rows):
            if v:
                if cur == 0:
                    start = y
                cur += 1
                if cur > best[1] - best[0]:
                    best = (start, y + 1)
            else:
                cur = 0
        out.append((a, b, best))
    # every panel in the strip has the same square size: cut the photo panel with the
    # geometry of its neighbour (its bright pixels can look like background)
    a1, b1, (y0, y1) = out[1]
    w = b1 - a1
    a0, b0, _ = out[0]
    gap = a1 - b1 + w  # unused, kept for clarity
    photo_left = a1 - int(round((a1 - a0 + (b0 - a0)) / 2)) if False else None
    cuts = []
    pitch = out[2][0] - out[1][0]
    left0 = a1 - pitch
    cuts.append(im[y0:y1, left0:left0 + w])
    for a, b, (ya, yb) in out[1:]:
        cuts.append(im[ya:yb, a:b])
    return [Image.fromarray(c.astype(np.uint8)) for c in cuts]


def heat_figure():
    rows = [
        (f"{S}/nbimg2/c33_5.png", "a brown dog is running through a grassy field",
         [(0, "photo"), (3, "dog"), (5, "running"), (9, "grassy"), (10, "field")]),
        (f"{S}/nbimg2/c33_4.png", "a man wearing a hat and sunglasses is wearing a red hat",
         [(0, "photo"), (2, "man"), (5, "hat"), (7, "sunglasses"), (9, "wearing")]),
    ]
    for r, (path, cap, picks) in enumerate(rows):
        ps = panels(path)
        fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.55))
        for k, ax in enumerate(axes.flat):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k < 5:
                i, word = picks[k]
                ax.imshow(ps[i])
                ax.set_title("original photo" if word == "photo" else f"word: \u201c{word}\u201d",
                             fontsize=11, fontweight="bold" if word != "photo" else "normal",
                             color="#C8102E" if word != "photo" else "#1A1A1A", pad=4)
            else:
                ax.axis("off")
                ax.text(0.05, 0.5, "Caption written by\nthe attention model:\n\n\u201c" + "\n".join(
                    __import__("textwrap").wrap(cap, 24)) + "\u201d", fontsize=10.5, va="center",
                    ha="left", color="#1A1A1A", style="italic", transform=ax.transAxes)
        fig.tight_layout(h_pad=0.6, w_pad=0.4)
        fig.savefig(f"{OUT}/fig_heatmap_{'ab'[r]}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def copy_png(src, name, crop=None, max_w=2400):
    im = Image.open(src).convert("RGB")
    if crop:
        im = im.crop(crop)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(f"{OUT}/{name}.png", optimize=True)
    print(name, im.size)


heat_figure()
