import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency

# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "lineplot_ano_vacinado.png")

GRUPO_LABELS = {0: "Não vacinados", 1: "Vacinados"}
GRUPO_ESTILO = {0: {"linestyle": "-", "marker": "o"},
                1: {"linestyle": "--", "marker": "s"}}

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
# 2. TESTE QUI-QUADRADO (associação entre vacinação e desfecho)
# ════════════════════════════════════════════════════════════
tab_chi2 = pd.crosstab(dados["Vacinado"], dados["Óbito"])
chi2, p_chi2, dof, _ = chi2_contingency(tab_chi2)
p_chi2_str = "p<0,001" if p_chi2 < 0.001 else f"p={p_chi2:.3f}".replace(".", ",")

# ════════════════════════════════════════════════════════════
# 3. GRÁFICO ÚNICO — todas as combinações grupo × desfecho
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
ax.set_facecolor(PANEL)
ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

for grupo in sorted(GRUPO_LABELS):
    sub = dados[dados["Vacinado"] == grupo]
    tab = sub.groupby(["Ano", "Óbito"]).size().unstack(fill_value=0)
    anos_com_dados = sorted(sub["Ano"].unique())
    tab = tab.reindex(anos_com_dados, fill_value=0)

    x = tab.index.tolist()
    alta = tab.get(0, pd.Series(0, index=tab.index))
    obito = tab.get(1, pd.Series(0, index=tab.index))
    estilo = GRUPO_ESTILO[grupo]

    ax.plot(x, alta, color=COR_ALTA, linewidth=2.5, markersize=8,
            markerfacecolor=COR_ALTA, markeredgecolor=BG, markeredgewidth=1.5,
            zorder=3, **estilo)
    ax.plot(x, obito, color=COR_OBITO, linewidth=2.5, markersize=8,
            markerfacecolor=COR_OBITO, markeredgecolor=BG, markeredgewidth=1.5,
            zorder=3, **estilo)

    dy_alta = 10 if grupo == 0 else 22
    dy_obito = -16 if grupo == 0 else -30
    for xi, a, o in zip(x, alta, obito):
        ax.annotate(str(a), (xi, a), textcoords="offset points",
                    xytext=(0, dy_alta), ha="center", fontsize=10,
                    color=COR_ALTA, fontweight="bold")
        ax.annotate(str(o), (xi, o), textcoords="offset points",
                    xytext=(0, dy_obito), ha="center", fontsize=10,
                    color=COR_OBITO, fontweight="bold")

ax.set_xticks(anos_todos)
ax.set_xticklabels([str(a) for a in anos_todos], fontsize=12, color=TEXT)
ax.set_xlim(min(anos_todos) - 0.3, max(anos_todos) + 0.3)
ax.tick_params(axis="y", colors=SUBTEXT, labelsize=11)
ax.tick_params(axis="x", length=0, pad=8)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylabel("Número de pacientes", fontsize=13, color=SUBTEXT, labelpad=8)
ymin, ymax = ax.get_ylim()
ax.set_ylim(-ymax * 0.05, ymax * 1.12)

# Legenda: cor = desfecho, estilo de linha/marcador = grupo de vacinação
legend_elements = [
    Line2D([0], [0], color=COR_ALTA, lw=2.5, label="Alta (Óbito = 0)"),
    Line2D([0], [0], color=COR_OBITO, lw=2.5, label="Óbito (Óbito = 1)"),
    Line2D([0], [0], color=TEXT, lw=2, linestyle="-", marker="o",
           label=f"Não vacinados (n={(dados['Vacinado'] == 0).sum()})"),
    Line2D([0], [0], color=TEXT, lw=2, linestyle="--", marker="s",
           label=f"Vacinados (n={(dados['Vacinado'] == 1).sum()})"),
]
ax.legend(handles=legend_elements, fontsize=11, frameon=True, edgecolor=BORDER,
          facecolor=BG, labelcolor=TEXT, loc="upper left", framealpha=0.97)

# Anotação do teste qui-quadrado (Vacinado × Óbito)
ax.text(0.99, 0.98, f"χ²={chi2:.2f}, gl={dof}, {p_chi2_str}",
        transform=ax.transAxes, fontsize=10, color="#555555",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                  alpha=0.95, linewidth=0.7))

fig.text(0.5, 0.985, "Altas e Óbitos por Ano — Vacinados vs. Não Vacinados",
          ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.5, 0.945, f"n total = {n_total} | Óbitos = {n_obitos}",
          ha="center", va="top", fontsize=12, color=SUBTEXT)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
plt.show()
