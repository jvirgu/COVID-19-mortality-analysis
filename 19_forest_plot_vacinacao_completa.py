import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Forest plot da vacinação completa (2+ doses) vs. não vacinados. Pacientes
# com apenas 1 dose são excluídos desta comparação: a vacinação parcial no
# contexto hospitalar costuma refletir vacinação tardia/emergencial durante
# a própria internação (viés de indicação), confundindo a associação.
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "forest_vacinacao_completa_vs_obito.png")

ADJUST_VARS = [
    "Sexo", "Prob_Card", "CP", "Diabetes", "SRAG", "Choques", "Prob_neurol",
    "Prob_Hemat", "Cancer", "Prob_Resp", "Prob_Metab", "Prob_TGI",
    "Prob_Hep", "Prob_Hid_Elet", "Prob_AI_Infla", "Febre", "Outros",
    "Traumatismo", "COVID_CRÍTICA", "Prob_Renal", "LRA", "Dias_permanência",
    "Estado_Civil_1", "Estado_Civil_2", "Idade_cat_1", "Idade_cat_2",
    "Idade_cat_3", "Grau_Instrucao_1", "Grau_Instrucao_2",
]

# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS E DEFINIR EXPOSIÇÃO (referência = não vacinado)
# ════════════════════════════════════════════════════════════
dados_todos = pd.read_excel(XLSX_PATH)
dados_todos["Completa"] = (dados_todos["Vacinas"] >= 2).astype(int)
n_excluidos_1dose = int((dados_todos["Vacinas"] == 1).sum())

dados = dados_todos[(dados_todos["Vacinas"] == 0) |
                     (dados_todos["Vacinas"] >= 2)].copy()

n_total_completo = len(dados_todos)
n_analisado = len(dados)
n_obitos = int(dados["Óbito"].sum())
n_completa = int(dados["Completa"].sum())
n_completa_obito = int(dados.loc[dados["Completa"] == 1, "Óbito"].sum())


def fit_or(formula, var, data):
    modelo = smf.glm(formula=formula, data=data,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    return np.exp(beta), np.exp(lo), np.exp(hi), p


OR, lo, hi, p = fit_or("Óbito ~ Completa", "Completa", dados)
formula_adj = "Óbito ~ Completa + " + " + ".join(ADJUST_VARS)
ORa, loa, hia, pa = fit_or(formula_adj, "Completa", dados)

df_raw = pd.DataFrame([{
    "label": "Vacinação completa (2+ doses)", "n_geral": n_completa,
    "n_obito": n_completa_obito, "OR": OR, "IC_inf": lo, "IC_sup": hi,
    "p_OR": p, "ORa": ORa, "ICa_inf": loa, "ICa_sup": hia, "p_ORa": pa,
}])

print("=" * 70)
print(f"Tabela lida de: {XLSX_PATH}")
print(f"n total = {n_total_completo} | analisados = {n_analisado} "
      f"(excluídos {n_excluidos_1dose} com 1 dose) | Óbitos = {n_obitos}")
print(df_raw.to_string())
print("=" * 70)

# ════════════════════════════════════════════════════════════
# 2. FOREST PLOT
# ════════════════════════════════════════════════════════════
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
GOLD = "#B08800"
COR_PROT = "#1D9E75"
COR_RISCO = "#E07B39"


def sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


n_rows = len(df_raw)
fig_h = max(4.5, n_rows * 0.9 + 3.0)
fig, axes = plt.subplots(
    1, 4, figsize=(16, fig_h), facecolor=BG,
    gridspec_kw={"width_ratios": [2.5, 5, 2.5, 5], "wspace": 0.04},
)
ax_labels, ax_or, ax_gap, ax_ora = axes
ax_gap.set_visible(False)

ax_labels.set_facecolor(BG)
ax_labels.set_xlim(0, 1)
ax_labels.set_ylim(n_rows - 0.5, -0.5)
for spine in ax_labels.spines.values():
    spine.set_visible(False)
ax_labels.set_xticks([])
ax_labels.set_yticks([])

for ax in (ax_labels, ax_or, ax_ora):
    for i in range(n_rows):
        bg_col = "#E8EDF2" if i % 2 == 0 else PANEL
        ax.axhspan(i - 0.42, i + 0.42, color=bg_col, alpha=0.55, zorder=0)

for i, row in df_raw.iterrows():
    ax_labels.text(0.98, i, f"{row['label']} (n={row['n_geral']})",
                   color=TEXT, fontsize=16, va="center", ha="right")


def draw_panel(ax, col_or, col_lo, col_hi, col_p, title, xlim):
    ax.set_facecolor(PANEL)
    ax.set_xscale("log")
    ax.xaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.axvline(1.0, color=GOLD, linewidth=2.5, linestyle="--", zorder=2, alpha=0.9)

    for i, row in df_raw.iterrows():
        OR_ = row[col_or]
        lo_ = row[col_lo]
        hi_ = row[col_hi]
        p_ = row[col_p]

        if (pd.isna(OR_) or pd.isna(lo_) or pd.isna(hi_)
                or not np.isfinite(OR_) or OR_ < 1e-6):
            ax.text(0.5, i, "—", color=SUBTEXT, fontsize=13,
                    va="center", ha="center",
                    transform=ax.get_yaxis_transform())
            continue

        cor = COR_PROT if OR_ < 1 else COR_RISCO
        sig = (p_ is not None) and not np.isnan(p_) and (p_ < 0.05)

        lo_plot = max(lo_, xlim[0] * 1.02)
        hi_plot = min(hi_, xlim[1] * 0.98) if np.isfinite(hi_) else xlim[1] * 0.98
        ax.plot([lo_plot, hi_plot], [i, i],
                color=cor, linewidth=3.8, zorder=3, alpha=0.85,
                solid_capstyle="round")

        ax.plot(OR_, i, marker="D" if sig else "o", markersize=9 if sig else 7,
                color=cor, markerfacecolor=cor if sig else BG,
                markeredgecolor=cor, markeredgewidth=1.6, zorder=5)

        hi_txt = min(hi_, 9999) if np.isfinite(hi_) else float("inf")
        hi_str = f"{hi_txt:.2f}" if np.isfinite(hi_txt) else "∞"
        txt = f"{OR_:.2f} ({lo_:.2f}–{hi_str}){sig_stars(p_)}"
        ax.text(1.02, i, txt, transform=ax.get_yaxis_transform(),
                color=TEXT, fontsize=15, va="center", ha="left", clip_on=False)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([""] * n_rows)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=14)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Odds Ratio (escala log)", fontsize=20, color=SUBTEXT, labelpad=6)
    ax.set_title(title, fontsize=20, fontweight="bold", color=TEXT, pad=5)


draw_panel(ax_or, "OR", "IC_inf", "IC_sup", "p_OR",
           "OR bruto (IC 95%)", xlim=(0.3, 3))
draw_panel(ax_ora, "ORa", "ICa_inf", "ICa_sup", "p_ORa",
           "OR ajustado (IC 95%)", xlim=(0.3, 3))

legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT,
                   label="Fator protetor (OR < 1)"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO,
                   label="Fator de risco (OR > 1)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT,
           markeredgecolor=TEXT, markersize=5, label="Losango = p < 0,05"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BG,
           markeredgecolor=TEXT, markeredgewidth=1.2, markersize=6,
           label="Círculo aberto = p ≥ 0,05"),
    Line2D([0], [0], color=GOLD, linewidth=1.2, linestyle="--",
           label="Linha de referência (OR = 1)"),
]
ax_or.legend(handles=legend_elements, fontsize=8, frameon=True, edgecolor=BORDER,
             facecolor=BG, labelcolor=TEXT, loc="lower left", framealpha=0.97,
             borderpad=0.9, handlelength=0.5)

fig.text(0.50, 1.14, "Forest Plot — Vacinação Completa vs. Óbito Hospitalar",
          ha="center", va="top", fontsize=30, fontweight="bold", color=TEXT)
subtitle = (f"Referência: não vacinados | Vacinação completa = 2+ doses | "
            f"n analisado = {n_analisado} (excluídos {n_excluidos_1dose} com "
            f"1 dose) | Óbitos = {n_obitos}")
fig.text(0.50, 1.02, subtitle, ha="center", va="top", fontsize=17, color=SUBTEXT)
fig.add_artist(plt.Line2D([0.13, 0.97], [0.96, 0.96], transform=fig.transFigure,
                           color=BORDER, linewidth=1.8))
fig.text(0.03, -0.16,
          "*** p<0,001 ** p<0,01 * p<0,05 | OR = Odds Ratio; IC = Intervalo "
          "de Confiança de 95% | Pacientes com 1 dose foram excluídos por "
          "possível viés de indicação (vacinação tardia/emergencial durante "
          "a internação) | Ajustado por sexo, comorbidades, idade, estado "
          "civil, escolaridade e tempo de internação",
          color=SUBTEXT, fontsize=11.5, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1.94])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
plt.show()
