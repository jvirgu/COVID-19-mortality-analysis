import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ─────────────────────────────────────────────────────────────────────────
# ANÁLISE LANDMARK — correção aproximada de viés de tempo imortal
#
# Limitação dos dados: a planilha não registra a DATA em que cada paciente
# foi vacinado, apenas o número de doses ao final da internação. Isso
# impede um modelo de Cox com covariável dependente do tempo "de verdade"
# (que exigiria saber, dia a dia, se o paciente já estava vacinado).
#
# Como proxy padrão na literatura, usa-se uma ANÁLISE LANDMARK: escolhe-se
# um ponto de corte precoce (aqui, o dia 3 de internação) e descartam-se os
# pacientes que morreram ou tiveram alta antes desse ponto. Como ninguém
# pode ser vacinado durante a internação E morrer antes do dia 3 no mesmo
# evento, isso remove a maior parte da distorção causada por vacinação
# tardia/emergencial (o viés de tempo imortal), sem precisar da data exata.
# A sobrevida é recontada a partir do landmark (dia 3) em diante.
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "landmark_vacinacao_completa.png")

LANDMARK_DIA = 3

BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_NAO_VAC = "#E0523F"
COR_COMPLETA = "#1D9E75"


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS, DEFINIR GRUPO E APLICAR O LANDMARK
# ════════════════════════════════════════════════════════════
dados_todos = pd.read_excel(XLSX_PATH)
dados_todos["Completa"] = (dados_todos["Vacinas"] >= 2).astype(int)
n_excluidos_1dose = int((dados_todos["Vacinas"] == 1).sum())

base = dados_todos[(dados_todos["Vacinas"] == 0) |
                    (dados_todos["Vacinas"] >= 2)].copy()
n_base = len(base)

# Exclui quem morreu ou teve alta antes do landmark (dia 3)
landmark = base[base["Dias_permanência"] >= LANDMARK_DIA].copy()
landmark["Tempo_landmark"] = landmark["Dias_permanência"] - LANDMARK_DIA
n_excluidos_landmark = n_base - len(landmark)

n_total = len(landmark)
n_obitos = int(landmark["Óbito"].sum())
n_completa = int(landmark["Completa"].sum())
n_completa_obito = int(landmark.loc[landmark["Completa"] == 1, "Óbito"].sum())
n_nao_vac = n_total - n_completa
n_nao_vac_obito = n_obitos - n_completa_obito

# ════════════════════════════════════════════════════════════
# 2. KAPLAN-MEIER + LOG-RANK NA COORTE LANDMARK
# ════════════════════════════════════════════════════════════
sub_nao = landmark[landmark["Completa"] == 0]
sub_completa = landmark[landmark["Completa"] == 1]

resultado_lr = logrank_test(sub_nao["Tempo_landmark"], sub_completa["Tempo_landmark"],
                             event_observed_A=sub_nao["Óbito"],
                             event_observed_B=sub_completa["Óbito"])
chi2_lr, p_lr = resultado_lr.test_statistic, resultado_lr.p_value

# ════════════════════════════════════════════════════════════
# 3. GRÁFICO
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG)
ax.set_facecolor(PANEL)
ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

kmf_nao = KaplanMeierFitter()
kmf_nao.fit(sub_nao["Tempo_landmark"], event_observed=sub_nao["Óbito"])
kmf_nao.plot_survival_function(ax=ax, color=COR_NAO_VAC, linewidth=2.6,
                                ci_alpha=0.15, show_censors=False, label=None)

kmf_completa = KaplanMeierFitter()
kmf_completa.fit(sub_completa["Tempo_landmark"], event_observed=sub_completa["Óbito"])
kmf_completa.plot_survival_function(ax=ax, color=COR_COMPLETA, linewidth=2.8,
                                     ci_alpha=0.15, show_censors=False, label=None)

ax.set_xlabel(f"Dias de internação após o landmark (dia {LANDMARK_DIA})",
              fontsize=13, color=SUBTEXT, labelpad=8)
ax.set_ylabel("Probabilidade de sobrevida (S(t))", fontsize=13, color=SUBTEXT,
              labelpad=8)
ax.set_ylim(0, 1.03)
ax.tick_params(axis="both", colors=SUBTEXT, labelsize=11)

legend_elements = [
    Line2D([0], [0], color=COR_COMPLETA, lw=2.8,
           label=f"Vacinação completa, 2+ doses (n={n_completa}, "
                 f"óbitos={n_completa_obito})"),
    Line2D([0], [0], color=COR_NAO_VAC, lw=2.6,
           label=f"Não vacinados (n={n_nao_vac}, óbitos={n_nao_vac_obito})"),
]
ax.legend(handles=legend_elements, fontsize=11, frameon=True, edgecolor=BORDER,
          facecolor=BG, labelcolor=TEXT, loc="lower left", framealpha=0.97)

ax.text(0.99, 0.98,
        f"Log-rank pós-landmark\nχ²={chi2_lr:.2f}, gl=1, {formata_p(p_lr)}",
        transform=ax.transAxes, fontsize=10.5, color="#333333",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray",
                  alpha=0.95, linewidth=0.8))

fig.text(0.5, 0.985,
          f"Análise Landmark (dia {LANDMARK_DIA}) — Sobrevida Pós-Landmark "
          "por Vacinação Completa",
          ha="center", va="top", fontsize=16.5, fontweight="bold", color=TEXT)
fig.text(0.5, 0.935,
          f"Coorte landmark: n = {n_total} (de {n_base} elegíveis; excluídos "
          f"{n_excluidos_landmark} óbitos/altas antes do dia {LANDMARK_DIA}) | "
          f"Óbitos pós-landmark = {n_obitos}",
          ha="center", va="top", fontsize=11, color=SUBTEXT)
nota_landmark = (
    "Correção aproximada de viés de tempo imortal: exclui pacientes que\n"
    f"morreram/tiveram alta antes do dia {LANDMARK_DIA}, já que nenhum deles poderia ter\n"
    "sido vacinado durante a internação e morrido tão cedo. A planilha não\n"
    "registra a data exata da vacinação, então esta é a alternativa padrão\n"
    "a um modelo de Cox com covariável dependente do tempo."
)
fig.text(0.13, -0.02, nota_landmark,
          ha="left", va="bottom", fontsize=9, style="italic", color=SUBTEXT)

plt.tight_layout(rect=[0, 0.20, 1, 0.90])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
print(f"Coorte base (0 ou 2+ doses, excluído 1 dose n={n_excluidos_1dose}): n={n_base}")
print(f"Excluídos pelo landmark (óbito/alta antes do dia {LANDMARK_DIA}): {n_excluidos_landmark}")
print(f"Coorte landmark: n={n_total}, óbitos={n_obitos}")
print(f"Log-rank pós-landmark: chi2={chi2_lr:.3f}, p={p_lr:.4f}")
plt.show()
