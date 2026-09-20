import os

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

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
# 2. TESTE QUI-QUADRADO — global e por ano (associação doses × desfecho)
# ════════════════════════════════════════════════════════════
tab_chi2 = pd.crosstab(dados["Vacinas"], dados["Óbito"])
chi2, p_chi2, dof, _ = chi2_contingency(tab_chi2)


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


p_chi2_str = formata_p(p_chi2)

# ════════════════════════════════════════════════════════════
# 2b. REGRESSÃO LOGÍSTICA POR ANO — OR de cada dose vs. referência
#     (nenhuma dose), ajustada separadamente nos dados de cada ano
# ════════════════════════════════════════════════════════════
texto_por_ano = {}
for ano in anos_todos:
    sub_ano = dados[dados["Ano"] == ano]
    doses_presentes = sorted(sub_ano["Vacinas"].unique())
    if len(doses_presentes) < 2:
        texto_por_ano[ano] = "Apenas um grupo\n(regressão não aplicável)"
        continue

    sub_dummies = pd.get_dummies(sub_ano, columns=["Vacinas"], dtype=int,
                                  drop_first=True)
    linhas = [f"Regressão logística {ano}\n(ref.: nenhuma dose)"]
    for dose in doses_presentes:
        if dose == 0:
            continue
        var = f"Vacinas_{dose}"
        if var not in sub_dummies.columns:
            continue
        modelo = smf.glm(f"Óbito ~ {var}", data=sub_dummies,
                          family=sm.families.Binomial()).fit()
        beta = modelo.params[var]
        p = modelo.pvalues[var]
        or_val = np.exp(beta)
        if or_val < 1e-6:
            linhas.append(f"{DOSE_LABELS[dose]}: OR não estimável")
        else:
            linhas.append(f"{DOSE_LABELS[dose]}: OR={or_val:.2f} ({formata_p(p)})")
    texto_por_ano[ano] = "\n".join(linhas)

# ════════════════════════════════════════════════════════════
# 3. GRÁFICO ÚNICO — todas as combinações dose × desfecho
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7.5), facecolor=BG)
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
ax.set_ylim(-ymax * 0.05, ymax * 1.55)

# Anotações da regressão logística por ano (OR vs. referência)
for ano in anos_todos:
    ax.text(ano, ymax * 1.51, texto_por_ano[ano], ha="center", va="top",
            fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray",
                      alpha=0.9, linewidth=0.8))

# Legenda (fora do painel, à direita): cor = desfecho, estilo = nº de doses
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
          facecolor=BG, labelcolor=TEXT, loc="center left",
          bbox_to_anchor=(1.01, 0.5), framealpha=0.97)

fig.text(0.44, 0.985, "Altas e Óbitos por Ano — Número de Doses de Vacina",
          ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.44, 0.945,
          f"n total = {n_total} | Óbitos = {n_obitos} | "
          f"χ² global={chi2:.2f}, gl={dof}, {p_chi2_str}",
          ha="center", va="top", fontsize=12, color=SUBTEXT)
fig.text(0.13, 0.02,
          "OR = razão de chances (Óbito ~ dose), regressão logística "
          "bruta ajustada separadamente em cada ano | "
          "algumas categorias têm n pequeno — interpretar com cautela",
          ha="left", va="bottom", fontsize=9, style="italic", color=SUBTEXT)

plt.tight_layout(rect=[0, 0.03, 0.83, 0.90])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
plt.show()
