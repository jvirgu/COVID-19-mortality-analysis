"""
Forest plot dos Odds Ratios do modelo de regressão logística ajustado
(multivariável) para Óbito, incluindo Vacinado — mesmo modelo estimado
em 05b_modelo_ajustado_vacinado.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "New_pacientes703.xlsx")


def _register_arial():
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            fm.fontManager.addfont(c)
            name = fm.FontProperties(fname=c).get_name()
            plt.rcParams["font.family"] = name
            return name
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


_register_arial()

# ════════════════════════════════════════════════════════════
# 1. MODEL (same as 05b_modelo_ajustado_vacinado.py)
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

dados["Idade_cat_1"] = (dados["Idade_cat"] == 1).astype(int)
dados["Idade_cat_2"] = (dados["Idade_cat"] == 2).astype(int)
dados["Idade_cat_3"] = (dados["Idade_cat"] == 3).astype(int)
dados["Estado_Civil_1"] = (dados["Estado_Civil"] == 1).astype(int)
dados["Estado_Civil_2"] = (dados["Estado_Civil"] == 2).astype(int)
dados["Grau_Instrucao_1"] = (dados["Grau de Instrução"] == 1).astype(float)
dados["Grau_Instrucao_2"] = (dados["Grau de Instrução"] == 2).astype(float)
dados.loc[dados["Grau de Instrução"].isna(), ["Grau_Instrucao_1", "Grau_Instrucao_2"]] = np.nan

formula = (
    "Óbito ~ Vacinado + Sexo + Prob_Card + CP + Município + Diabetes + "
    "SRAG + Choques + Prob_neurol + Prob_Hemat + Cancer + Prob_Resp + "
    "Prob_Metab + Prob_TGI + Prob_Hep + Prob_Hid_Elet + Prob_AI_Infla + "
    "Febre + Outros + Traumatismo + COVID_CRÍTICA + Prob_Renal + LRA + "
    "Dias_permanência + Estado_Civil_1 + Estado_Civil_2 + Idade_cat_1 + "
    "Idade_cat_2 + Idade_cat_3 + Grau_Instrucao_1 + Grau_Instrucao_2"
)
modelo = smf.glm(formula=formula, data=dados, family=sm.families.Binomial()).fit()
n_model = int(modelo.nobs)

LABELS = {
    "Vacinado": "Vacinado",
    "Sexo": "Sexo",
    "Prob_Card": "Problema cardíaco",
    "CP": "CP",
    "Município": "Município (cat. 1)",
    "Diabetes": "Diabetes",
    "SRAG": "SRAG",
    "Choques": "Choque",
    "Prob_neurol": "Problema neurológico",
    "Prob_Hemat": "Problema hematológico",
    "Cancer": "Câncer",
    "Prob_Resp": "Problema respiratório",
    "Prob_Metab": "Problema metabólico",
    "Prob_TGI": "Problema gastrointestinal",
    "Prob_Hep": "Problema hepático",
    "Prob_Hid_Elet": "Distúrbio hidroeletrolítico",
    "Prob_AI_Infla": "Doença autoimune/inflamatória",
    "Febre": "Febre",
    "Outros": "Outras comorbidades",
    "Traumatismo": "Traumatismo",
    "COVID_CRÍTICA": "COVID crítica",
    "Prob_Renal": "Problema renal",
    "LRA": "Lesão renal aguda",
    "Dias_permanência": "Dias de internação (por dia)",
    "Estado_Civil_1": "Estado civil (cat. 1 vs 0)",
    "Estado_Civil_2": "Estado civil (cat. 2 vs 0)",
    "Idade_cat_1": "Idade 19-40 (vs 0-18)",
    "Idade_cat_2": "Idade 41-60 (vs 0-18)",
    "Idade_cat_3": "Idade +60 (vs 0-18)",
    "Grau_Instrucao_1": "Escolaridade (cat. 1 vs 0)",
    "Grau_Instrucao_2": "Escolaridade (cat. 2 vs 0)",
}

or_values = np.exp(modelo.params)
conf = modelo.conf_int()
tab = pd.DataFrame({
    "OR": or_values,
    "IC_inf": np.exp(conf[0]),
    "IC_sup": np.exp(conf[1]),
    "p": modelo.pvalues,
}).drop(index="Intercept")
tab["label"] = [LABELS.get(v, v) for v in tab.index]
tab = tab.sort_values("OR", ascending=True).reset_index(drop=True)
n_rows = len(tab)

# ════════════════════════════════════════════════════════════
# 2. FOREST PLOT (style consistent with 08_forest_plot_critical_covid.py)
# ════════════════════════════════════════════════════════════
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
GOLD = "#B08800"
COR_PROT = "#1D9E75"   # OR < 1 -> protective
COR_RISCO = "#E07B39"  # OR > 1 -> risk


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


fig_h = max(6, n_rows * 0.55 + 2.5)
fig, (ax_labels, ax_or) = plt.subplots(
    1, 2, figsize=(13, fig_h), facecolor=BG,
    gridspec_kw={"width_ratios": [2.6, 5], "wspace": 0.03},
)

ax_labels.set_facecolor(BG)
ax_labels.set_xlim(0, 1)
ax_labels.set_ylim(-0.5, n_rows - 0.5)
for spine in ax_labels.spines.values():
    spine.set_visible(False)
ax_labels.set_xticks([])
ax_labels.set_yticks([])

for ax in (ax_labels, ax_or):
    for i in range(n_rows):
        bg_col = "#E8EDF2" if i % 2 == 0 else PANEL
        ax.axhspan(i - 0.42, i + 0.42, color=bg_col, alpha=0.55, zorder=0)

for i, row in tab.iterrows():
    weight = "bold" if row["label"] == "Vacinado" else "normal"
    ax_labels.text(0.98, i, row["label"], color=TEXT, fontsize=11,
                   va="center", ha="right", fontweight=weight)

xlim = (tab["IC_inf"].replace(0, np.nan).min() * 0.6,
        min(tab["IC_sup"].max(), 1e4) * 1.6)

ax_or.set_facecolor(PANEL)
ax_or.set_xscale("log")
ax_or.xaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax_or.set_axisbelow(True)
for spine in ax_or.spines.values():
    spine.set_edgecolor(BORDER)
    spine.set_linewidth(0.8)
ax_or.axvline(1.0, color=GOLD, linewidth=2.2, linestyle="--", zorder=2, alpha=0.9)

for i, row in tab.iterrows():
    OR, lo, hi, p = row["OR"], row["IC_inf"], row["IC_sup"], row["p"]
    cor = COR_PROT if OR < 1 else COR_RISCO
    sig = p < 0.05
    lo_plot = max(lo, xlim[0] * 1.02)
    hi_plot = min(hi, xlim[1] * 0.98)
    ax_or.plot([lo_plot, hi_plot], [i, i], color=cor, linewidth=3.2, zorder=3,
               alpha=0.85, solid_capstyle="round")
    ax_or.plot(OR, i, marker="D" if sig else "o", markersize=8 if sig else 6.5,
               color=cor, markerfacecolor=cor if sig else BG,
               markeredgecolor=cor, markeredgewidth=1.5, zorder=5)
    hi_txt = min(hi, 9999)
    txt = f"{OR:.2f} ({lo:.2f}–{hi_txt:.2f}){sig_stars(p)}"
    ax_or.text(1.02, i, txt, transform=ax_or.get_yaxis_transform(),
               color=TEXT, fontsize=10.5, va="center", ha="left", clip_on=False)

ax_or.set_yticks(range(n_rows))
ax_or.set_yticklabels([""] * n_rows)
ax_or.tick_params(axis="y", length=0)
ax_or.tick_params(axis="x", colors=SUBTEXT, labelsize=10.5)
ax_or.set_ylim(-0.5, n_rows - 0.5)
ax_or.set_xlim(*xlim)
ax_or.set_xlabel("Odds Ratio (escala log)", fontsize=12, color=SUBTEXT, labelpad=6)

legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT, label="Fator protetor (OR < 1)"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO, label="Fator de risco (OR > 1)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT, markeredgecolor=TEXT,
           markersize=6, label="Losango = p < 0,05"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BG, markeredgecolor=TEXT,
           markeredgewidth=1.2, markersize=7, label="Círculo aberto = p ≥ 0,05"),
    Line2D([0], [0], color=GOLD, linewidth=1.4, linestyle="--", label="Referência (OR = 1)"),
]
ax_or.legend(handles=legend_elements, fontsize=8.5, frameon=True, edgecolor=BORDER,
             facecolor=BG, labelcolor=TEXT, loc="upper left", bbox_to_anchor=(1.35, 1.0),
             framealpha=0.97, borderpad=0.9, handlelength=1.2)

fig.suptitle("Forest Plot — Odds Ratios ajustados para Óbito (inclui Vacinado)",
             fontsize=15.5, fontweight="bold", color=TEXT, y=1.02)
fig.text(0.5, 0.985, f"Regressão logística multivariável (n = {n_model}); IC 95%",
         ha="center", va="top", fontsize=10, color=SUBTEXT)

fig.subplots_adjust(top=0.92, bottom=0.07, left=0.26, right=0.78)

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(BASE_DIR, f"forest_plot_modelo_ajustado.{ext}")
    fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
