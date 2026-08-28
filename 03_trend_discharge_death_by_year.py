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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paleta (mesma do dashboard) ──────────────────────────────────────────────
ORANGE = "#E07B39"
TEAL = "#1D9E75"
BG = "#f5f4f0"
SURF = "#ffffff"
MUTED = "#6b6b63"
TEXT = "#1a1a18"
GRID = "#e0ddd8"

# ── Preparar dados ────────────────────────────────────────────────────────
df1['Data de Entrada'] = pd.to_datetime(df1['Data de Entrada'], dayfirst=True)
df1['Ano'] = df1['Data de Entrada'].dt.year
trend = (
    df1.groupby(['Ano', 'Óbito'])
    .size()
    .reset_index(name='Contagem')
)
obito = trend[trend['Óbito'] == 1].set_index('Ano')['Contagem']
alta = trend[trend['Óbito'] == 0].set_index('Ano')['Contagem']
anos = sorted(trend['Ano'].unique())

# ── Figura ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
ax.set_facecolor(SURF)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color(GRID)

# ── Linhas e área ─────────────────────────────────────────────────────────
ax.fill_between(alta.index, alta.values, alpha=0.15, color=TEAL)
ax.fill_between(obito.index, obito.values, alpha=0.15, color=ORANGE)
ax.plot(alta.index, alta.values, color=TEAL, linewidth=2.5,
        marker='o', markersize=10, markerfacecolor=SURF,
        markeredgecolor=TEAL, markeredgewidth=2, label='Alta (0)')
ax.plot(obito.index, obito.values, color=ORANGE, linewidth=2.5,
        marker='o', markersize=10, markerfacecolor=SURF,
        markeredgecolor=ORANGE, markeredgewidth=2, label='Óbito (1)')

# ── Anotações nos pontos ──────────────────────────────────────────────────
for serie, cor in [(alta, TEAL), (obito, ORANGE)]:
    for ano, val in serie.items():
        if ano == 2021:
            offset_x = -15 if cor == TEAL else 15
            offset_y = 10
            ha = 'right' if cor == TEAL else 'left'
        else:
            offset_x = 0
            offset_y = 10
            ha = 'center'
        ax.annotate(
            str(val),
            xy=(ano, val),
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            ha=ha, va='bottom',
            fontsize=15, color=cor,
            fontfamily='Arial', fontweight='bold',
        )

# ── Eixos ─────────────────────────────────────────────────────────────────
ax.set_xticks(anos)
ax.set_xticklabels([str(a) for a in anos], fontsize=18,
                    color=MUTED, fontfamily='Arial')
# tick labels do eixo Y — tick_params não aceita fontfamily, usar loop
ax.tick_params(axis='y', labelsize=12, colors=MUTED)
for lbl in ax.get_yticklabels():
    lbl.set_fontfamily('Arial')
ax.set_xlabel('Year of admission', fontsize=18, color=MUTED,
              fontfamily='Arial', labelpad=12)
ax.set_ylabel('Number of patients', fontsize=18, color=MUTED,
              fontfamily='Arial', labelpad=12)
ax.grid(axis='y', color=GRID, linewidth=0.6, linestyle='--')
ax.set_axisbelow(True)

# ── Título e legenda ──────────────────────────────────────────────────────
ax.set_title('TREND OF DISCHARGE AND DEATH BY YEAR',
              fontsize=18, fontweight='bold', color=TEXT,
              fontfamily='Arial', loc='left', pad=15)
legend_handles = [
    mpatches.Patch(facecolor=TEAL + "66", edgecolor=TEAL, label='Discharge (0)'),
    mpatches.Patch(facecolor=ORANGE + "66", edgecolor=ORANGE, label='Death (1)'),
]
ax.legend(handles=legend_handles, frameon=False,
          loc='upper left', handlelength=1.2,
          prop={'family': 'Arial', 'size': 15})

plt.tight_layout()
plt.savefig('tendencia_obito_alta.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.show()
