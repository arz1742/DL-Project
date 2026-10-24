"""Redraw the t-SNE plot from the Colab output image (cell 17): dot centres are read back from the
saved PNG by colour, then re-plotted larger with fewer groups so the message is easy to see."""
import sys, os, glob
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__))
for f in glob.glob(os.path.join(HERE, "fonts", "OpenSans-*.ttf")): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Open Sans"], "font.size": 11})
im = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
H, W, _ = im.shape
tab = {"other": (199, 199, 199), "dog": (31, 119, 180), "man": (255, 127, 14), "boy": (44, 160, 44),
       "girl": (214, 39, 40), "woman": (148, 103, 189), "people": (140, 86, 75), "child": (227, 119, 194),
       "player": (127, 127, 127)}
# plot area: inside the axes frame, excluding the legend box (top right)
mask_area = np.zeros((H, W), bool); mask_area[58:1016, 24:1148] = True
mask_area[52:310, 940:1145] = False
pts = {}
for name, c in tab.items():
    d = np.abs(im - np.array(c)).sum(-1)
    m = (d < 60) & mask_area
    lab, n = ndimage.label(m)
    if n == 0: continue
    idx = range(1, n + 1)
    areas = ndimage.sum(m, lab, idx); cents = ndimage.center_of_mass(m, lab, idx)
    single = np.median(areas[areas > 10]) if (areas > 10).any() else 30
    out = []
    for a, (cy, cx) in zip(areas, cents):
        if a < 8: continue
        k = max(1, int(round(a / single)))
        for j in range(k):
            out.append((cx + (j - (k - 1) / 2) * 2.0, -cy))
    pts[name] = np.array(out)
    print(name, len(out))
TRUE_N = {"Other / unclear": 178, "Dog": 212, "Sports player": 54}  # counts from the Colab legend
TRUE_N["People (man, woman, boy, girl, child, people)"] = 194 + 87 + 80 + 74 + 62 + 59
groups = [("Other / unclear", ["other"], "#CFCFCF", 10),
          ("People (man, woman, boy, girl, child, people)", ["man", "woman", "boy", "girl", "child", "people"], "#1A1A1A", 12),
          ("Sports player", ["player"], "#E8737F", 14),
          ("Dog", ["dog"], "#C8102E", 14)]
fig, ax = plt.subplots(figsize=(6.6, 5.0))
for label, keys, col, s in groups:
    P = np.vstack([pts[k] for k in keys if k in pts and len(pts[k])])
    ax.scatter(P[:, 0], P[:, 1], s=s, c=col, lw=0, alpha=0.9, label=f"{label}: {TRUE_N[label]} images")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color("#BBBBBB")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=10.5, markerscale=2.0)
ax.annotate("Dogs form their own cluster", xy=(1000, -640), xytext=(830, -200), fontsize=11, color="#C8102E",
            fontweight="bold", arrowprops=dict(arrowstyle="->", color="#C8102E"))
ax.annotate("All kinds of people are mixed together", xy=(260, -600), xytext=(30, -1010), fontsize=11, color="#1A1A1A",
            fontweight="bold", arrowprops=dict(arrowstyle="->", color="#1A1A1A"))
ax.set_ylim(-1060, -40)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig_tsne.pdf"), bbox_inches="tight", pad_inches=0.03)
