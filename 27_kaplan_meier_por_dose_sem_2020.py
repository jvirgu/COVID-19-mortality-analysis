import os

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# ─────────────────────────────────────────────────────────────────────────
# Curvas de Kaplan-Meier durante a internação, uma por número de doses de
# vacina (0 a 4), EXCLUINDO pacientes internados em 2020 — ano anterior ao
# início da vacinação, em que todos os pacientes são necessariamente "0
# doses" (essas internações inflam e distorcem o grupo de referência).
# Tempo = dias de permanência hospitalar; evento = óbito (alta = censura).
# Teste de log-rank multivariado (k-amostras) para associação global entre
# nº de doses e sobrevida.
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "kaplan_meier_por_dose_sem_2020.png")

DOSE_LABELS = {0: "0 doses", 1: "1 dose", 2: "2 doses",
               3: "3 doses", 4: "4 doses"}
# Estilo por dose: cor sequencial (mais doses = mais escuro) + estilos de
# linha distintos para leitura em P&B
CORES = {0: "#E0523F", 1: "#E0A23F", 2: "#8FA83A", 3: "#3F9E7A", 4: "#1D6B9E"}
ESTILOS = {0: "-", 1: "--", 2: ":", 3: "-.", 4: (0, (3, 1, 1, 1))}

BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS (excluindo 2020)
# ════════════════════════════════════════════════════════════
dados_full = pd.read_excel(XLSX_PATH)
dados_full["Ano"] = pd.to_datetime(dados_full["Data de Entrada"]).dt.year
n_excluidos_2020 = int((dados_full["Ano"] == 2020).sum())
dados = dados_full[dados_full["Ano"] != 2020].copy()

n_total = len(dados)
n_obitos = int(dados["Óbito"].sum())
doses = sorted(dados["Vacinas"].unique())

# Teste de log-rank multivariado (k-amostras) — associação global
resultado_lr = multivariate_logrank_test(
    dados["Dias_permanência"], dados["Vacinas"], dados["Óbito"])
chi2_lr, p_lr, gl_lr = (resultado_lr.test_statistic, resultado_lr.p_value,
                         resultado_lr.degrees_of_freedom)

# ════════════════════════════════════════════════════════════
# 2. GRÁFICO — uma curva de KM por número de doses
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG)
ax.set_facecolor(PANEL)
ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

for dose in doses:
    sub = dados[dados["Vacinas"] == dose]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["Dias_permanência"], event_observed=sub["Óbito"])
    kmf.plot_survival_function(
        ax=ax, color=CORES[dose], linewidth=2.4, linestyle=ESTILOS[dose],
        ci_show=False, show_censors=False,
        label=f"{DOSE_LABELS[dose]} (n={len(sub)}, óbitos={int(sub['Óbito'].sum())})")

ax.set_xlabel("Dias de internação", fontsize=13, color=SUBTEXT, labelpad=8)
ax.set_ylabel("Probabilidade de sobrevida (S(t))", fontsize=13, color=SUBTEXT,
              labelpad=8)
ax.set_ylim(0, 1.03)
xmax = dados["Dias_permanência"].quantile(0.98)
ax.set_xlim(0, xmax)
ax.tick_params(axis="both", colors=SUBTEXT, labelsize=11)

ax.legend(fontsize=10.5, frameon=True, edgecolor=BORDER, facecolor=BG,
          labelcolor=TEXT, loc="lower left", framealpha=0.97)

ax.text(0.99, 0.98,
        f"Log-rank multivariado (0-4 doses)\nχ²={chi2_lr:.2f}, gl={gl_lr}, "
        f"{formata_p(p_lr)}",
        transform=ax.transAxes, fontsize=10.5, color="#333333",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray",
                  alpha=0.95, linewidth=0.8))

fig.text(0.5, 0.985,
          "Sobrevida Durante a Internação por Número de Doses de Vacina "
          "(excluindo 2020)",
          ha="center", va="top", fontsize=17, fontweight="bold", color=TEXT)
fig.text(0.5, 0.945,
          f"n = {n_total} (excluídos {n_excluidos_2020} de 2020) | "
          f"Óbitos = {n_obitos} | Tempo = dias de permanência | "
          "evento = óbito (alta = censura)",
          ha="center", va="top", fontsize=11.5, color=SUBTEXT)

plt.tight_layout(rect=[0, 0.02, 1, 0.92])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
print(f"n = {n_total} (excluídos {n_excluidos_2020} de 2020)")
print(f"Log-rank multivariado (0-4 doses): chi2={chi2_lr:.3f}, gl={gl_lr}, p={p_lr:.4f}")
plt.show()
