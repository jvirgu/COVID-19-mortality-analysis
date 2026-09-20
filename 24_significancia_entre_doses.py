import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import fisher_exact

# ─────────────────────────────────────────────────────────────────────────
# Significância estatística ENTRE todas as combinações de número de doses
# (não apenas cada dose vs. referência). Usa o teste exato de Fisher, mais
# apropriado que o qui-quadrado para tabelas 2x2 com células pequenas.
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

# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS E CALCULAR TODAS AS COMPARAÇÕES PAREADAS
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)
tab = pd.crosstab(dados["Vacinas"], dados["Óbito"]).reindex(
    sorted(DOSE_LABELS), fill_value=0)
tab.columns = ["Alta", "Obito"]
doses = sorted(DOSE_LABELS)

resultados = []
matriz_p = pd.DataFrame(np.nan, index=doses, columns=doses)
for a, b in combinations(doses, 2):
    sub = tab.loc[[a, b]]
    odds, p = fisher_exact(sub)
    matriz_p.loc[a, b] = p
    matriz_p.loc[b, a] = p
    resultados.append({
        "Dose A": DOSE_LABELS[a], "Dose B": DOSE_LABELS[b],
        "OR (Fisher)": odds, "p-valor": p,
        "n A": int(tab.loc[a].sum()), "n B": int(tab.loc[b].sum()),
    })

df_resultados = pd.DataFrame(resultados)
df_resultados["Significativo (p<0,05)"] = df_resultados["p-valor"] < 0.05

print("=" * 90)
print("Comparações pareadas entre doses (teste exato de Fisher, tabela 2x2 de Óbito)")
print(df_resultados.to_string(index=False))
print("=" * 90)


def formata_p(p):
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


# ════════════════════════════════════════════════════════════
# 2. HEATMAP DA MATRIZ DE P-VALORES (triângulo inferior)
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor=BG)

n = len(doses)
cmap = LinearSegmentedColormap.from_list(
    "sig", ["#1D9E75", "#E8EDF2", "#E8EDF2"])  # verde forte perto de 0

# Máscara: mostra apenas o triângulo inferior (evita duplicar cada par)
display_vals = matriz_p.values.copy()
mask = np.triu(np.ones_like(display_vals, dtype=bool), k=0)
display_vals_masked = np.ma.array(display_vals, mask=mask)

im = ax.imshow(display_vals_masked, cmap="Greens_r", vmin=0, vmax=0.15,
               aspect="equal")

for i in range(n):
    for j in range(n):
        if i <= j:
            continue
        p = matriz_p.iloc[i, j]
        sig = p < 0.05
        cor_txt = "white" if p < 0.05 else TEXT
        peso = "bold" if sig else "normal"
        marca = "*" if sig else ""
        ax.text(j, i, f"p={formata_p(p)}{marca}", ha="center", va="center",
                color=cor_txt, fontsize=11, fontweight=peso)

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

fig.text(0.5, 0.98, "Significância Estatística Entre Doses de Vacina",
          ha="center", va="top", fontsize=17, fontweight="bold", color=TEXT)
fig.text(0.5, 0.935,
          "Teste exato de Fisher (tabela 2x2 de Óbito) para cada par de "
          "grupos de dose",
          ha="center", va="top", fontsize=11.5, color=SUBTEXT)
fig.text(0.5, 0.02, "* p < 0,05 (diferença estatisticamente significativa)",
          ha="center", va="bottom", fontsize=10.5, style="italic", color=SUBTEXT)

plt.tight_layout(rect=[0, 0.05, 1, 0.90])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"\nGráfico salvo em: {OUTPUT_PNG}")
plt.show()
