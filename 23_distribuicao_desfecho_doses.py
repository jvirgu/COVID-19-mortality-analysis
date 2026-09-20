import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "distribuicao_desfecho_doses.png")

DOSE_LABELS = {0: "0 doses", 1: "1 dose", 2: "2 doses",
               3: "3 doses", 4: "4 doses"}

# ── Paleta ────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_ALTA = "#1D9E75"
COR_OBITO = "#E0703A"

# ════════════════════════════════════════════════════════════
# 1. CARREGAR E AGREGAR OS DADOS (percentual dentro de cada grupo de dose)
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)
tab = pd.crosstab(dados["Vacinas"], dados["Óbito"]).reindex(
    sorted(DOSE_LABELS), fill_value=0)
tab.columns = ["Alta", "Obito"]
n_grupo = tab.sum(axis=1)
pct = tab.div(n_grupo, axis=0) * 100

x = list(range(len(tab)))
labels = [DOSE_LABELS[d] for d in tab.index]

# ════════════════════════════════════════════════════════════
# 2. GRÁFICO
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
ax.set_facecolor(PANEL)
ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.fill_between(x, pct["Alta"], color=COR_ALTA, alpha=0.18, zorder=1)
ax.plot(x, pct["Alta"], color=COR_ALTA, linewidth=2.6, marker="o",
        markersize=8, markerfacecolor=COR_ALTA, markeredgecolor=BG,
        markeredgewidth=1.4, zorder=3, label="Alta (0)")

ax.fill_between(x, pct["Obito"], color=COR_OBITO, alpha=0.18, zorder=1,
                linestyle="--")
ax.plot(x, pct["Obito"], color=COR_OBITO, linewidth=2.6, linestyle="--",
        marker="o", markersize=8, markerfacecolor=COR_OBITO,
        markeredgecolor=BG, markeredgewidth=1.4, zorder=3, label="Óbito (1)")

# Rótulos de percentual e n em cada ponto
for xi, dose in zip(x, tab.index):
    ax.annotate(f"{pct.loc[dose, 'Alta']:.1f}%\n(n={tab.loc[dose, 'Alta']})",
                (xi, pct.loc[dose, "Alta"]), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=10.5, color=TEXT)
    ax.annotate(f"{pct.loc[dose, 'Obito']:.1f}%\n(n={tab.loc[dose, 'Obito']})",
                (xi, pct.loc[dose, "Obito"]), textcoords="offset points",
                xytext=(0, -30), ha="center", fontsize=10.5, color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13, color=TEXT)
ax.set_xlim(-0.3, len(x) - 0.7)
ax.set_ylim(-8, 118)
ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.tick_params(axis="y", colors=SUBTEXT, labelsize=11)
ax.tick_params(axis="x", length=0, pad=8)
ax.set_xlabel("Número de doses de vacina", fontsize=13, color=TEXT, labelpad=10)
ax.set_ylabel("% dentro do grupo de dose", fontsize=13, color=TEXT, labelpad=8)

ax.legend(fontsize=11.5, frameon=True, edgecolor=BORDER, facecolor=BG,
          labelcolor=TEXT, loc="upper left", framealpha=0.95)

fig.text(0.5, 0.98, "Distribuição do Desfecho por Número de Doses de Vacina",
          ha="center", va="top", fontsize=19, fontweight="bold", color=TEXT)
fig.text(0.5, 0.02, "Percentual calculado dentro de cada grupo de dose",
          ha="center", va="bottom", fontsize=12, color=SUBTEXT)

plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
print(tab)
print(pct.round(1))
plt.show()
