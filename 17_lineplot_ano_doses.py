import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "lineplot_ano_doses.png")

DOSE_LABELS = {0: "Nenhuma dose", 1: "1 dose", 2: "2 doses",
               3: "3 doses", 4: "4 doses"}
# Estilo de linha/marcador por número de doses (a cor identifica o desfecho)
DOSE_ESTILO = {
    0: {"linestyle": "-", "marker": "o"},
    1: {"linestyle": "--", "marker": "s"},
    2: {"linestyle": ":", "marker": "^"},
    3: {"linestyle": "-.", "marker": "D"},
    4: {"linestyle": (0, (3, 1, 1, 1)), "marker": "v"},
}

# ── Paleta ────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_ALTA = "#1D9E75"   # alta (desfecho favorável)
COR_OBITO = "#E0523F"  # óbito (desfecho crítico)

# ════════════════════════════════════════════════════════════
# 1. CARREGAR E AGREGAR OS DADOS
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)
dados["Ano"] = pd.to_datetime(dados["Data de Entrada"]).dt.year

n_total = len(dados)
n_obitos = int(dados["Óbito"].sum())
anos_todos = sorted(dados["Ano"].unique())

# ════════════════════════════════════════════════════════════
# 2. GRÁFICO ÚNICO — todas as combinações dose × desfecho
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG)
ax.set_facecolor(PANEL)
ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

for dose in sorted(DOSE_LABELS):
    sub = dados[dados["Vacinas"] == dose]
    tab = sub.groupby(["Ano", "Óbito"]).size().unstack(fill_value=0)
    anos_com_dados = sorted(sub["Ano"].unique())
    tab = tab.reindex(anos_com_dados, fill_value=0)

    x = tab.index.tolist()
    alta = tab.get(0, pd.Series(0, index=tab.index))
    obito = tab.get(1, pd.Series(0, index=tab.index))
    estilo = DOSE_ESTILO[dose]

    ax.plot(x, alta, color=COR_ALTA, linewidth=2.2, markersize=7,
            markerfacecolor=COR_ALTA, markeredgecolor=BG, markeredgewidth=1.3,
            zorder=3, alpha=0.9, **estilo)
    ax.plot(x, obito, color=COR_OBITO, linewidth=2.2, markersize=7,
            markerfacecolor=COR_OBITO, markeredgecolor=BG, markeredgewidth=1.3,
            zorder=3, alpha=0.9, **estilo)

    # Rótulos apenas no último ponto de cada série (10 séries deixariam o
    # gráfico ilegível se todos os pontos fossem anotados)
    ax.annotate(str(alta.iloc[-1]), (x[-1], alta.iloc[-1]),
                textcoords="offset points", xytext=(8, 0), va="center",
                ha="left", fontsize=9, color=COR_ALTA, fontweight="bold")
    ax.annotate(str(obito.iloc[-1]), (x[-1], obito.iloc[-1]),
                textcoords="offset points", xytext=(8, 0), va="center",
                ha="left", fontsize=9, color=COR_OBITO, fontweight="bold")

ax.set_xticks(anos_todos)
ax.set_xticklabels([str(a) for a in anos_todos], fontsize=12, color=TEXT)
ax.set_xlim(min(anos_todos) - 0.3, max(anos_todos) + 0.6)
ax.tick_params(axis="y", colors=SUBTEXT, labelsize=11)
ax.tick_params(axis="x", length=0, pad=8)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylabel("Número de pacientes", fontsize=13, color=SUBTEXT, labelpad=8)
ymin, ymax = ax.get_ylim()
ax.set_ylim(-ymax * 0.05, ymax * 1.15)

# Legenda: cor = desfecho, estilo de linha/marcador = número de doses
legend_elements = [
    Line2D([0], [0], color=COR_ALTA, lw=2.5, label="Alta (Óbito = 0)"),
    Line2D([0], [0], color=COR_OBITO, lw=2.5, label="Óbito (Óbito = 1)"),
]
for dose in sorted(DOSE_LABELS):
    n = int((dados["Vacinas"] == dose).sum())
    legend_elements.append(
        Line2D([0], [0], color=TEXT, lw=1.8, label=f"{DOSE_LABELS[dose]} (n={n})",
               **DOSE_ESTILO[dose]))

ax.legend(handles=legend_elements, fontsize=10, frameon=True, edgecolor=BORDER,
          facecolor=BG, labelcolor=TEXT, loc="upper left", framealpha=0.97,
          ncol=1)

fig.text(0.5, 0.985, "Altas e Óbitos por Ano — Número de Doses de Vacina",
          ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.5, 0.95, f"n total = {n_total} | Óbitos = {n_obitos}",
          ha="center", va="top", fontsize=12, color=SUBTEXT)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
plt.show()
