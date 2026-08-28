import os
import matplotlib as mpl
import matplotlib.font_manager as fm


# ── Registrar Arial (detecta automaticamente no sistema) ─────────────────────
def _registrar_arial():
    fontes = [f.name for f in fm.fontManager.ttflist]
    if "Arial" in fontes:
        return
    candidatos = [
        r"C:\Windows\Fonts\arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        os.path.join(os.path.dirname(__file__), "Arial.ttf"),
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            fm.fontManager.addfont(caminho)
            return
    import urllib.request
    dest = os.path.join(os.path.expanduser("~"), "Arial.ttf")
    if not os.path.exists(dest):
        url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
        urllib.request.urlretrieve(url, dest)
    fm.fontManager.addfont(dest)


_registrar_arial()
mpl.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ── Estilo ────────────────────────────────────────────────────────────────
sns.set_style("white")
sns.set_context("talk")
mpl.rcParams["font.family"] = "Arial"  # reafirmar após set_context (seaborn sobrescreve)

# ── Dados ─────────────────────────────────────────────────────────────────
tabela = df1["Óbito"].value_counts().to_frame().T

# ── Mapa de cores ─────────────────────────────────────────────────────────
cores = LinearSegmentedColormap.from_list(
    "verde_laranja",
    ["#FFCB99", "#ffe29a", "#A8D8C8"],
)

# ── Figura ────────────────────────────────────────────────────────────────
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
        "fontfamily": "Arial",  # anotações dentro das células
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
cbar.ax.yaxis.label.set_fontfamily("Arial")  # rótulo da colorbar
cbar.ax.yaxis.label.set_fontsize(18)
for lbl in cbar.ax.get_yticklabels():  # tick labels da colorbar
    lbl.set_fontfamily("Arial")

# ── Fundo ─────────────────────────────────────────────────────────────────
ax.set_facecolor("#f5f5f5")

# ── Título ────────────────────────────────────────────────────────────────
plt.title(
    "Distribution of Outcome",
    fontsize=25,
    weight="bold",
    pad=20,
    fontfamily="Arial",
)

# ── Eixos ─────────────────────────────────────────────────────────────────
plt.xlabel("Death or Discharge", fontsize=16, weight="bold", fontfamily="Arial")
plt.ylabel("", fontfamily="Arial")
plt.xticks(fontsize=13, weight="bold")
plt.yticks(rotation=0)
for lbl in ax.get_xticklabels():
    lbl.set_fontfamily("Arial")
for lbl in ax.get_yticklabels():
    lbl.set_fontfamily("Arial")

# ── Layout & Salvar ───────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig("heatmap_obito_alta.png", dpi=180, bbox_inches="tight")
plt.show()
