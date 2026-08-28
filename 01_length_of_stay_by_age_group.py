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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── Estilo ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")
mpl.rcParams["font.family"] = "Arial"  # reafirmar após set_theme (seaborn sobrescreve)

# ── Faixas etárias ────────────────────────────────────────────────────────
bins = [0, 18, 40, 60, 120]
labels = ['0-18', '19-40', '41-60', '+60']
df1['Faixa_étaria'] = pd.cut(df1['Idade'], bins=bins, labels=labels)

# ── Figura ────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 7))
ax = sns.boxplot(
    x='Faixa_étaria',
    y='Dias_permanência',
    hue='Óbito',
    data=df1,
    palette={0: '#38b000', 1: '#ff9f43'},
    linewidth=2,
    fliersize=3,
)
for patch in ax.patches:
    patch.set_alpha(0.75)

# ── Legenda ───────────────────────────────────────────────────────────────
handles, _ = plt.gca().get_legend_handles_labels()
leg = plt.legend(
    handles,
    ['Discharge', 'Death'],
    title='Outcome',
    prop={'family': 'Arial', 'size': 16},
)
leg.get_title().set_fontfamily('Arial')
leg.get_title().set_fontsize(19)

# ── Título e labels ───────────────────────────────────────────────────────
plt.title(
    'Length of Stay by Age Group',
    fontsize=25,
    weight='bold',
    fontfamily='Arial',
)
plt.xlabel('Age Group', fontsize=18, fontfamily='Arial')
plt.ylabel('Length of stay', fontsize=18, fontfamily='Arial')

# ── Tick labels ───────────────────────────────────────────────────────────
for lbl in ax.get_xticklabels():
    lbl.set_fontfamily('Arial')
for lbl in ax.get_yticklabels():
    lbl.set_fontfamily('Arial')

# ── Finalizar ─────────────────────────────────────────────────────────────
sns.despine()
plt.tight_layout()
plt.savefig('boxplot_dias_faixa_etaria.png', dpi=180, bbox_inches='tight')
plt.show()
