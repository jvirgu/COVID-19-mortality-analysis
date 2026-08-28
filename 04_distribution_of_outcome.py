import os
import matplotlib as mpl
import matplotlib.font_manager as fm


# ── Register Arial (auto-detects on the system) ──────────────────────────────
def _register_arial():
    fonts = [f.name for f in fm.fontManager.ttflist]
    if "Arial" in fonts:
        return
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        os.path.join(os.path.dirname(__file__), "Arial.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            return
    import urllib.request
    dest = os.path.join(os.path.expanduser("~"), "Arial.ttf")
    if not os.path.exists(dest):
        url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
        urllib.request.urlretrieve(url, dest)
    fm.fontManager.addfont(dest)


_register_arial()
mpl.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_style("white")
sns.set_context("talk")
mpl.rcParams["font.family"] = "Arial"  # reassert after set_context (seaborn overrides it)

# ── Data ──────────────────────────────────────────────────────────────────
tabela = df1["Óbito"].value_counts().to_frame().T

# ── Color map ─────────────────────────────────────────────────────────────
cores = LinearSegmentedColormap.from_list(
    "green_orange",
    ["#FFCB99", "#ffe29a", "#A8D8C8"],
)

# ── Figure ────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    tabela,
    annot=True,
    fmt="d", cmap=cores,
    linewidths=2,
    linecolor="white",
    annot_kws={
        "size": 18,
        "weight": "bold",
        "color": "black",
        "fontfamily": "Arial",  # annotations inside the cells
    },
    cbar_kws={
        "label": "Number of patients",
        "shrink": 0.8,
    },
)

# ── Colorbar ──────────────────────────────────────────────────────────────
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_rotation(270)
cbar.ax.yaxis.set_label_coords(5.5, 0.5)
cbar.ax.yaxis.label.set_fontfamily("Arial")  # colorbar label
cbar.ax.yaxis.label.set_fontsize(18)
for lbl in cbar.ax.get_yticklabels():  # colorbar tick labels
    lbl.set_fontfamily("Arial")

# ── Background ────────────────────────────────────────────────────────────
ax.set_facecolor("#f5f5f5")

# ── Title ─────────────────────────────────────────────────────────────────
plt.title(
    "Distribution of Outcome",
    fontsize=25,
    weight="bold",
    pad=20,
    fontfamily="Arial",
)

# ── Axes ──────────────────────────────────────────────────────────────────
plt.xlabel("Death or Discharge", fontsize=16, weight="bold", fontfamily="Arial")
plt.ylabel("", fontfamily="Arial")
plt.xticks(fontsize=13, weight="bold")
plt.yticks(rotation=0)
for lbl in ax.get_xticklabels():
    lbl.set_fontfamily("Arial")
for lbl in ax.get_yticklabels():
    lbl.set_fontfamily("Arial")

# ── Layout & save ─────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig("heatmap_obito_alta.png", dpi=180, bbox_inches="tight")
plt.show()
