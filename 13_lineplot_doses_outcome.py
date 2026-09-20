import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "lineplot_doses_outcome.png")

DOSE_LABELS = {0: "Nenhuma dose", 1: "1 dose", 2: "2 doses",
               3: "3 doses", 4: "4 doses"}

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
tab = pd.crosstab(dados["Vacinas"], dados["Óbito"]).reindex(
    sorted(DOSE_LABELS), fill_value=0)
tab.columns = ["Alta", "Obito"]

x = list(range(len(tab)))
labels = [DOSE_LABELS[d] for d in tab.index]
n_total = len(dados)
n_obitos = int(dados["Óbito"].sum())

# ════════════════════════════════════════════════════════════
# 2. ANÁLISE ESTATÍSTICA POR DOSE
# ════════════════════════════════════════════════════════════
# Teste qui-quadrado global (associação entre nº de doses e desfecho)
chi2, p_chi2, dof, _ = chi2_contingency(tab[["Alta", "Obito"]])

# OR bruto de cada dose vs. referência (nenhuma dose), via regressão
# logística simples (Óbito ~ dose), mesma abordagem do forest plot de doses
dados_dummies = pd.get_dummies(dados, columns=["Vacinas"], dtype=int,
                                drop_first=True)
or_por_dose = {0: None}  # referência não tem OR
for dose in [1, 2, 3, 4]:
    var = f"Vacinas_{dose}"
    modelo = smf.glm(f"Óbito ~ {var}", data=dados_dummies,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    or_val, lo_val, hi_val = np.exp(beta), np.exp(lo), np.exp(hi)
    or_por_dose[dose] = (or_val, lo_val, hi_val, p)


def formata_or(item):
    if item is None:
        return "Referência"
    or_val, lo_val, hi_val, p = item
    if or_val < 1e-6:
        return "OR não estimável"
    hi_str = f"{min(hi_val, 999):.2f}" if np.isfinite(hi_val) else "∞"
    p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    return f"OR={or_val:.2f} [{lo_val:.2f}–{hi_str}]\n{p_str}"


# ════════════════════════════════════════════════════════════
# 3. GRÁFICO DE LINHAS
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11.5, 7.5), facecolor=BG)
ax.set_facecolor(PANEL)

ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.plot(x, tab["Alta"], color=COR_ALTA, linewidth=2.5, marker="o",
        markersize=8, markerfacecolor=COR_ALTA, markeredgecolor=BG,
        markeredgewidth=1.5, zorder=3, label="Alta (Óbito = 0)")
ax.plot(x, tab["Obito"], color=COR_OBITO, linewidth=2.5, marker="o",
        markersize=8, markerfacecolor=COR_OBITO, markeredgecolor=BG,
        markeredgewidth=1.5, zorder=3, label="Óbito (Óbito = 1)")

# Rótulos diretos de valor acima/abaixo de cada ponto
for xi, (a, o) in zip(x, zip(tab["Alta"], tab["Obito"])):
    ax.annotate(str(a), (xi, a), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=11, color=COR_ALTA, fontweight="bold")
    ax.annotate(str(o), (xi, o), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=11, color=COR_OBITO, fontweight="bold")

# Anotações de OR bruto (vs. referência) por dose
y_max = tab.values.max()
for xi, dose in zip(x, tab.index):
    ax.text(xi, y_max * 1.22, formata_or(or_por_dose[dose]),
            ha="center", va="top", fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                      alpha=0.9, linewidth=0.8))

ax.set_xticks(x)
ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, tab.sum(axis=1))],
                    fontsize=12, color=TEXT)
ax.tick_params(axis="y", colors=SUBTEXT, labelsize=11)
ax.tick_params(axis="x", length=0, pad=12)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylabel("Número de pacientes", fontsize=13, color=SUBTEXT, labelpad=8)
ax.set_ylim(-y_max * 0.05, y_max * 1.75)
ax.set_xlim(-0.5, len(x) - 0.5)

ax.legend(fontsize=11, frameon=True, edgecolor=BORDER, facecolor=BG,
          labelcolor=TEXT, loc="lower center", framealpha=0.97,
          bbox_to_anchor=(0.5, 1.0), ncol=2)

# Anotação do teste qui-quadrado global
p_chi2_str = "p<0,001" if p_chi2 < 0.001 else f"p={p_chi2:.3f}".replace(".", ",")
ax.text(0.01, 0.98, f"χ²={chi2:.2f}, gl={dof}, {p_chi2_str}",
        transform=ax.transAxes, fontsize=10, color="#555555",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                  alpha=0.95, linewidth=0.7))

fig.text(0.5, 0.985, "Altas e Óbitos por Número de Doses de Vacina",
          ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.5, 0.955,
          f"n total = {n_total} | Óbitos = {n_obitos} | "
          "OR bruto vs. referência (nenhuma dose)",
          ha="center", va="top", fontsize=12, color=SUBTEXT)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
print(tab)
print(f"\nQui-quadrado global: chi2={chi2:.3f}, gl={dof}, p={p_chi2:.4f}")
for dose, item in or_por_dose.items():
    print(f"Dose {dose}: {item}")
plt.show()
