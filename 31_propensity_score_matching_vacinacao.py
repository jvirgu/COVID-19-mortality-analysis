import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# PAREAMENTO POR ESCORE DE PROPENSÃO (PSM) — vacinação completa (2+ doses)
# vs. não vacinados. Em vez de ajustar um modelo logístico com ~28
# covariáveis para uma amostra pequena (74 vacinados), o escore de
# propensão resume essas covariáveis em um único número (a probabilidade
# prevista de estar vacinado, dadas as características basais) e pareia
# cada vacinado ao não vacinado mais parecido nesse escore. Isso reduz o
# desbalanceamento entre os grupos sem sofrer com o overfitting de um
# modelo ajustado direto com poucos eventos.
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "propensity_score_matching_vacinacao.png")

RANDOM_STATE = 42
CALIPER_FRACAO = 0.2  # caliper = 0.2 * desvio-padrão do logit(PS), padrão da literatura

# Covariáveis BASAIS (pré-tratamento) usadas para estimar o escore de
# propensão. Excluímos Dias_permanência (ocorre após/durante a exposição,
# não é uma característica basal) e os grupos de idade categorizados
# (redundantes com Idade contínua).
COVARIAVEIS = [
    "Sexo", "Idade", "Prob_Card", "CP", "Diabetes", "SRAG", "Choques",
    "Prob_neurol", "Prob_Hemat", "Cancer", "Prob_Resp", "Prob_Metab",
    "Prob_TGI", "Prob_Hep", "Prob_Hid_Elet", "Prob_AI_Infla", "Febre",
    "Outros", "Traumatismo", "COVID_CRÍTICA", "Prob_Renal", "LRA",
    "Estado_Civil_1", "Estado_Civil_2", "Grau_Instrucao_1", "Grau_Instrucao_2",
]

NOMES_COVARIAVEIS = {
    "Sexo": "Sexo", "Idade": "Idade", "Prob_Card": "Doença cardíaca",
    "CP": "Comprometimento pulmonar", "Diabetes": "Diabetes", "SRAG": "SRAG",
    "Choques": "Choque", "Prob_neurol": "Doença neurológica",
    "Prob_Hemat": "Doença hematológica", "Cancer": "Câncer",
    "Prob_Resp": "Doença respiratória", "Prob_Metab": "Doença metabólica",
    "Prob_TGI": "Doença gastrointestinal", "Prob_Hep": "Doença hepática",
    "Prob_Hid_Elet": "Distúrbio hidroeletrolítico",
    "Prob_AI_Infla": "Doença autoimune/inflamatória", "Febre": "Febre",
    "Outros": "Outras comorbidades", "Traumatismo": "Traumatismo",
    "COVID_CRÍTICA": "COVID crítica", "Prob_Renal": "Doença renal",
    "LRA": "Lesão renal aguda", "Estado_Civil_1": "Estado civil (cat. 1)",
    "Estado_Civil_2": "Estado civil (cat. 2)",
    "Grau_Instrucao_1": "Escolaridade (cat. 1)",
    "Grau_Instrucao_2": "Escolaridade (cat. 2)",
}

BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_ANTES = "#9AA3AC"
COR_DEPOIS = "#1D9E75"
COR_LIMITE = "#B08800"


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


def smd(x_trat, x_ctrl):
    """Diferença padronizada de médias (Cohen's d) entre dois grupos."""
    n1, n2 = len(x_trat), len(x_ctrl)
    var_pool = ((n1 - 1) * x_trat.var(ddof=1) + (n2 - 1) * x_ctrl.var(ddof=1)) / (n1 + n2 - 2)
    if var_pool == 0:
        return 0.0
    return (x_trat.mean() - x_ctrl.mean()) / np.sqrt(var_pool)


# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS E DEFINIR TRATAMENTO (referência = não vacinado)
# ════════════════════════════════════════════════════════════
dados_todos = pd.read_excel(XLSX_PATH)
n_excluidos_1dose = int((dados_todos["Vacinas"] == 1).sum())
dados = dados_todos[(dados_todos["Vacinas"] == 0) |
                     (dados_todos["Vacinas"] >= 2)].copy().reset_index(drop=True)
dados["Completa"] = (dados["Vacinas"] >= 2).astype(int)

# ════════════════════════════════════════════════════════════
# 2. ESTIMAR O ESCORE DE PROPENSÃO
# ════════════════════════════════════════════════════════════
formula = "Completa ~ " + " + ".join(COVARIAVEIS)
modelo_ps = smf.glm(formula, data=dados, family=sm.families.Binomial()).fit()
dados["ps"] = modelo_ps.predict(dados)
eps = 1e-6
ps_clip = dados["ps"].clip(eps, 1 - eps)
dados["logit_ps"] = np.log(ps_clip / (1 - ps_clip))

# ════════════════════════════════════════════════════════════
# 3. PAREAMENTO 1:1 POR VIZINHO MAIS PRÓXIMO, SEM REPOSIÇÃO, COM CALIPER
# ════════════════════════════════════════════════════════════
caliper = CALIPER_FRACAO * dados["logit_ps"].std()
tratados = dados[dados["Completa"] == 1].sample(frac=1, random_state=RANDOM_STATE)
controles = dados[dados["Completa"] == 0].copy()

pares = []
usados = set()
for idx, linha in tratados.iterrows():
    candidatos = controles[~controles.index.isin(usados)]
    dif = (candidatos["logit_ps"] - linha["logit_ps"]).abs()
    if dif.empty:
        continue
    melhor_idx = dif.idxmin()
    if dif.loc[melhor_idx] <= caliper:
        pares.append((idx, melhor_idx))
        usados.add(melhor_idx)

idx_tratados_pareados = [p[0] for p in pares]
idx_controles_pareados = [p[1] for p in pares]
pareados = dados.loc[idx_tratados_pareados + idx_controles_pareados]

n_tratados = len(tratados)
n_pareados = len(pares)

# ════════════════════════════════════════════════════════════
# 4. BALANCEAMENTO DAS COVARIÁVEIS (ANTES x DEPOIS DO PAREAMENTO)
# ════════════════════════════════════════════════════════════
smd_antes = {}
smd_depois = {}
for var in COVARIAVEIS:
    smd_antes[var] = smd(dados.loc[dados["Completa"] == 1, var],
                          dados.loc[dados["Completa"] == 0, var])
    smd_depois[var] = smd(pareados.loc[pareados["Completa"] == 1, var],
                           pareados.loc[pareados["Completa"] == 0, var])

df_balanco = pd.DataFrame({
    "Variável": [NOMES_COVARIAVEIS[v] for v in COVARIAVEIS],
    "SMD antes": [smd_antes[v] for v in COVARIAVEIS],
    "SMD depois": [smd_depois[v] for v in COVARIAVEIS],
})
df_balanco = df_balanco.reindex(
    df_balanco["SMD antes"].abs().sort_values(ascending=True).index)

# ════════════════════════════════════════════════════════════
# 5. DESFECHO NA AMOSTRA PAREADA
# ════════════════════════════════════════════════════════════
tab_pareada = pd.crosstab(pareados["Completa"], pareados["Óbito"])
chi2_pareado, p_pareado, dof_pareado, _ = chi2_contingency(tab_pareada)

or_pareado_num = (tab_pareada.loc[1, 1] * tab_pareada.loc[0, 0])
or_pareado_den = (tab_pareada.loc[1, 0] * tab_pareada.loc[0, 1])
or_pareado = or_pareado_num / or_pareado_den if or_pareado_den > 0 else np.nan

print("=" * 70)
print(f"Coorte base (0 ou 2+ doses, excluídos {n_excluidos_1dose} com 1 dose): {len(dados)}")
print(f"Pareados: {n_pareados} de {n_tratados} vacinados completos (caliper={caliper:.3f})")
print(tab_pareada)
print(f"OR pareado: {or_pareado:.3f} | chi2={chi2_pareado:.3f}, p={p_pareado:.4f}")
print("=" * 70)

# ════════════════════════════════════════════════════════════
# 6. GRÁFICO — LOVE PLOT (balanceamento) + DESFECHO PAREADO
# ════════════════════════════════════════════════════════════
fig, (ax_love, ax_desfecho) = plt.subplots(
    1, 2, figsize=(16, 8.5), facecolor=BG,
    gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.35})

# --- Love plot ---
ax_love.set_facecolor(PANEL)
ax_love.xaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax_love.set_axisbelow(True)
for spine in ax_love.spines.values():
    spine.set_visible(False)

y_pos = np.arange(len(df_balanco))
ax_love.axvline(0.1, color=COR_LIMITE, linewidth=1.6, linestyle="--", zorder=2)
ax_love.axvline(-0.1, color=COR_LIMITE, linewidth=1.6, linestyle="--", zorder=2)
ax_love.axvline(0, color=BORDER, linewidth=1.2, zorder=1)

ax_love.scatter(df_balanco["SMD antes"], y_pos, color=COR_ANTES, s=70,
                 marker="o", label="Antes do pareamento", zorder=3)
ax_love.scatter(df_balanco["SMD depois"], y_pos, color=COR_DEPOIS, s=70,
                 marker="D", label="Depois do pareamento", zorder=4)

ax_love.set_yticks(y_pos)
ax_love.set_yticklabels(df_balanco["Variável"], fontsize=10.5, color=TEXT)
ax_love.set_xlabel("Diferença padronizada de médias (SMD)", fontsize=12,
                    color=SUBTEXT, labelpad=8)
ax_love.tick_params(axis="x", colors=SUBTEXT, labelsize=10.5)
ax_love.set_title("Balanceamento das covariáveis (Love plot)", fontsize=14.5,
                   fontweight="bold", color=TEXT, pad=10)
ax_love.legend(fontsize=10, frameon=True, edgecolor=BORDER, facecolor=BG,
               labelcolor=TEXT, loc="lower right", framealpha=0.95)
xmax_love = max(0.35, df_balanco[["SMD antes", "SMD depois"]].abs().max().max() * 1.15)
ax_love.set_xlim(-xmax_love, xmax_love)

# --- Desfecho na amostra pareada ---
ax_desfecho.set_facecolor(PANEL)
ax_desfecho.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax_desfecho.set_axisbelow(True)
for spine in ax_desfecho.spines.values():
    spine.set_visible(False)

pct_obito = pareados.groupby("Completa")["Óbito"].mean() * 100
cores_barra = ["#E0703A", "#1D9E75"]
ax_desfecho.bar(["Não vacinados\n(pareados)", "Vacinação\ncompleta"],
                 [pct_obito.get(0, 0), pct_obito.get(1, 0)],
                 color=cores_barra, width=0.55, zorder=3)
for i, grupo in enumerate([0, 1]):
    n_grupo = int((pareados["Completa"] == grupo).sum())
    obitos_grupo = int(pareados.loc[pareados["Completa"] == grupo, "Óbito"].sum())
    ax_desfecho.text(i, pct_obito.get(grupo, 0) + 1.5,
                      f"{pct_obito.get(grupo, 0):.1f}%\n(n={n_grupo}, "
                      f"óbitos={obitos_grupo})",
                      ha="center", va="bottom", fontsize=10, color=TEXT)

ax_desfecho.set_ylabel("% de óbito", fontsize=12, color=SUBTEXT, labelpad=8)
ax_desfecho.set_ylim(0, max(pct_obito) * 1.62)
ax_desfecho.tick_params(axis="both", colors=SUBTEXT, labelsize=10.5)
ax_desfecho.set_title("Óbito na amostra pareada", fontsize=14.5,
                       fontweight="bold", color=TEXT, pad=10)
ax_desfecho.text(0.5, 0.99,
                  f"OR pareado = {or_pareado:.2f} | χ²={chi2_pareado:.2f}, "
                  f"gl={dof_pareado}, {formata_p(p_pareado)}",
                  transform=ax_desfecho.transAxes, ha="center", va="top",
                  fontsize=10, color=TEXT,
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray",
                            alpha=0.95, linewidth=0.8))

fig.text(0.5, 0.99,
          "Pareamento por Escore de Propensão — Vacinação Completa vs. "
          "Não Vacinados",
          ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.5, 0.945,
          f"{n_pareados} de {n_tratados} vacinados completos pareados 1:1 "
          f"(caliper = 0,2 DP do logit do escore) | "
          f"amostra pareada final: n = {2 * n_pareados}",
          ha="center", va="top", fontsize=12, color=SUBTEXT)
fig.text(0.045, 0.02,
          "SMD (diferença padronizada de médias) dentro de ±0,1 (linhas "
          "douradas) indica bom balanceamento entre os grupos após o "
          "pareamento.",
          ha="left", va="bottom", fontsize=9.5, style="italic", color=SUBTEXT)

plt.tight_layout(rect=[0, 0.06, 1, 0.90])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
plt.show()
