import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Significância estatística ENTRE todas as combinações de número de doses
# (não apenas cada dose vs. referência). Usa regressão logística (Óbito ~
# dose), a mesma abordagem dos forest plots de dose (scripts 12/13), em vez
# do teste exato de Fisher: cada par produz OR, IC 95% e p-valor.
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "significancia_entre_doses.png")

DOSE_LABELS = {0: "0 doses", 1: "1 dose", 2: "2 doses",
               3: "3 doses", 4: "4 doses"}

BG = "#FFFFFF"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
BORDER = "#D0D7DE"


def formata_p(p):
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS E CALCULAR TODAS AS COMPARAÇÕES PAREADAS
#    (regressão logística: Óbito ~ dose, referência = dose A)
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)
tab = pd.crosstab(dados["Vacinas"], dados["Óbito"]).reindex(
    sorted(DOSE_LABELS), fill_value=0)
tab.columns = ["Alta", "Obito"]
doses = sorted(DOSE_LABELS)

resultados = []
matriz_p = pd.DataFrame(np.nan, index=doses, columns=doses)
matriz_or = pd.DataFrame(np.nan, index=doses, columns=doses)
matriz_texto = pd.DataFrame("", index=doses, columns=doses)

for a, b in combinations(doses, 2):
    sub = dados[dados["Vacinas"].isin([a, b])].copy()
    sub["Exposta"] = (sub["Vacinas"] == b).astype(int)

    n_a, obitos_a = len(sub[sub["Vacinas"] == a]), int(sub.loc[sub["Vacinas"] == a, "Óbito"].sum())
    n_b, obitos_b = len(sub[sub["Vacinas"] == b]), int(sub.loc[sub["Vacinas"] == b, "Óbito"].sum())

    try:
        modelo = smf.glm("Óbito ~ Exposta", data=sub,
                          family=sm.families.Binomial()).fit()
        beta = modelo.params["Exposta"]
        p = modelo.pvalues["Exposta"]
        lo, hi = modelo.conf_int().loc["Exposta"]
        odds, odds_lo, odds_hi = np.exp(beta), np.exp(lo), np.exp(hi)
        estimavel = np.isfinite(odds) and odds > 1e-6
    except Exception:
        odds = odds_lo = odds_hi = p = np.nan
        estimavel = False

    if not estimavel:
        # Separação quase completa (ex.: grupo com 0 óbitos) deixa a
        # regressão logística instável (p≈1 não confiável). Nesses casos,
        # usa-se o teste exato de Fisher só para o p-valor/significância.
        tab_par = pd.crosstab(sub["Vacinas"], sub["Óbito"])
        _, p = fisher_exact(tab_par)
        matriz_texto.loc[b, a] = f"OR não estimável\n(Fisher p={formata_p(p)})"
    else:
        hi_str = f"{min(odds_hi, 999):.2f}" if np.isfinite(odds_hi) else "∞"
        matriz_texto.loc[b, a] = (f"OR={odds:.2f}\n[{odds_lo:.2f}–{hi_str}]\n"
                                   f"p={formata_p(p)}")

    matriz_p.loc[b, a] = p
    matriz_or.loc[b, a] = odds

    resultados.append({
        "Dose A (ref.)": DOSE_LABELS[a], "Dose B": DOSE_LABELS[b],
        "n A": n_a, "óbitos A": obitos_a, "n B": n_b, "óbitos B": obitos_b,
        "OR": odds, "IC 95% inf": odds_lo, "IC 95% sup": odds_hi, "p-valor": p,
    })

df_resultados = pd.DataFrame(resultados)
df_resultados["Significativo (p<0,05)"] = df_resultados["p-valor"] < 0.05

print("=" * 100)
print("Comparações pareadas entre doses (regressão logística: Óbito ~ dose)")
print(df_resultados.to_string(index=False))
print("=" * 100)

# ════════════════════════════════════════════════════════════
# 2. HEATMAP DA MATRIZ DE P-VALORES (triângulo inferior)
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9.5, 8.5), facecolor=BG)

n = len(doses)
display_vals = matriz_p.values.copy()
mask = np.triu(np.ones_like(display_vals, dtype=bool), k=0)
display_vals_masked = np.ma.array(display_vals, mask=mask)

ax.imshow(display_vals_masked, cmap="Greens_r", vmin=0, vmax=0.15, aspect="equal")

for i in range(n):
    for j in range(n):
        if i <= j:
            continue
        p = matriz_p.iloc[i, j]
        if np.isnan(p):
            continue
        sig = p < 0.05
        cor_txt = "white" if sig else TEXT
        peso = "bold" if sig else "normal"
        texto = matriz_texto.iloc[i, j]
        if sig:
            texto += "*"
        ax.text(j, i, texto, ha="center", va="center",
                color=cor_txt, fontsize=9.5, fontweight=peso, linespacing=1.4)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([DOSE_LABELS[d] for d in doses], fontsize=11, color=TEXT)
ax.set_yticklabels([DOSE_LABELS[d] for d in doses], fontsize=11, color=TEXT)
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color=BORDER, linewidth=1.2)
ax.tick_params(which="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# Oculta as células do triângulo superior (não usadas)
for i in range(n):
    for j in range(n):
        if i <= j:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor=BG, edgecolor=BG, zorder=5))

ax.set_xlabel("Referência (dose A)", fontsize=12, color=SUBTEXT, labelpad=10)
ax.set_ylabel("Exposta (dose B)", fontsize=12, color=SUBTEXT, labelpad=10)

fig.text(0.5, 0.98, "Significância Estatística Entre Doses de Vacina",
          ha="center", va="top", fontsize=17, fontweight="bold", color=TEXT)
fig.text(0.5, 0.935,
          "Regressão logística (Óbito ~ dose) para cada par de grupos — "
          "OR, IC 95% e p-valor",
          ha="center", va="top", fontsize=11.5, color=SUBTEXT)
fig.text(0.5, 0.02, "* p < 0,05 (diferença estatisticamente significativa)",
          ha="center", va="bottom", fontsize=10.5, style="italic", color=SUBTEXT)

plt.tight_layout(rect=[0, 0.05, 1, 0.90])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"\nGráfico salvo em: {OUTPUT_PNG}")
plt.show()
