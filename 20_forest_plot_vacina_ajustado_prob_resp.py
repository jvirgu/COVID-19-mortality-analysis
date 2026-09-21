"""
Modelo de regressão logística ajustado para Óbito, com Vacinado
controlado pela presença de problema respiratório (Prob_Resp) —
Óbito ~ Vacinado + Prob_Resp — apresentado como forest plot.

Objetivo: verificar se o efeito de Vacinado sobre o óbito muda quando
se ajusta apenas pelo confundidor mais relevante identificado nas
análises anteriores (Prob_Resp), de forma mais direta e interpretável
que o modelo completo de 30 variáveis (05b_modelo_ajustado_vacinado.py
/ 14_forest_plot_modelo_ajustado.py).
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
# 1. MODEL: Óbito ~ Vacinado + Prob_Resp
# ════════════════════════════════════════════════════════════
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
modelo = smf.glm("Óbito ~ Vacinado + Prob_Resp", data=df, family=sm.families.Binomial()).fit()
print(modelo.summary())
n_model = int(modelo.nobs)

LABELS = {"Vacinado": "Vacinado", "Prob_Resp": "Problema respiratório"}

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

print("\n=== Odds Ratios (ajustado: Vacinado + Prob_Resp) ===")
pd.set_option("display.float_format", "{:.4f}".format)
print(tab.to_string())

# ════════════════════════════════════════════════════════════
# 2. FOREST PLOT (mesmo estilo de 14_forest_plot_modelo_ajustado.py)
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
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


fig, (ax_labels, ax_or) = plt.subplots(
    1, 2, figsize=(13, 3.4), facecolor=BG,
    gridspec_kw={"width_ratios": [2.0, 4], "wspace": 0.03},
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
    ax_labels.text(0.98, i, row["label"], color=TEXT, fontsize=12, va="center", ha="right")

xlim = (tab["IC_inf"].min() * 0.7, tab["IC_sup"].max() * 1.5)

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
    ax_or.plot([lo, hi], [i, i], color=cor, linewidth=3.5, zorder=3, alpha=0.85, solid_capstyle="round")
    ax_or.plot(OR, i, marker="D" if sig else "o", markersize=10 if sig else 8,
               color=cor, markerfacecolor=cor if sig else BG,
               markeredgecolor=cor, markeredgewidth=1.6, zorder=5)
    txt = f"{OR:.2f} ({lo:.2f}–{hi:.2f}){sig_stars(p)}"
    ax_or.text(1.02, i, txt, transform=ax_or.get_yaxis_transform(),
               color=TEXT, fontsize=12, va="center", ha="left", clip_on=False)

ax_or.set_yticks(range(n_rows))
ax_or.set_yticklabels([""] * n_rows)
ax_or.tick_params(axis="y", length=0)
ax_or.set_ylim(-0.5, n_rows - 0.5)
ax_or.set_xlim(*xlim)

from matplotlib.ticker import FixedLocator, FixedFormatter
ticks = [t for t in [0.5, 1, 2, 3, 4, 5] if xlim[0] <= t <= xlim[1]]
ax_or.xaxis.set_minor_locator(FixedLocator([]))
ax_or.xaxis.set_major_locator(FixedLocator(ticks))
ax_or.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
ax_or.tick_params(axis="x", colors=SUBTEXT, labelsize=11)
ax_or.set_xlabel("Odds Ratio (escala log)", fontsize=12, color=SUBTEXT, labelpad=6)

legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT, label="Fator protetor (OR < 1)"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO, label="Fator de risco (OR > 1)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT, markeredgecolor=TEXT,
           markersize=7, label="Losango = p < 0,05"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BG, markeredgecolor=TEXT,
           markeredgewidth=1.2, markersize=8, label="Círculo aberto = p ≥ 0,05"),
    Line2D([0], [0], color=GOLD, linewidth=1.4, linestyle="--", label="Referência (OR = 1)"),
]
fig.legend(handles=legend_elements, fontsize=9, frameon=True, edgecolor=BORDER,
           facecolor=BG, labelcolor=TEXT, loc="center left", bbox_to_anchor=(0.99, 0.5),
           framealpha=0.97, borderpad=0.9, handlelength=1.2)

fig.suptitle("Forest Plot — Vacinado ajustado por Problema Respiratório",
             fontsize=14.5, fontweight="bold", color=TEXT, y=1.14)
fig.text(0.5, 1.045, f"Óbito ~ Vacinado + Prob_Resp (regressão logística; n = {n_model}); IC 95%",
         ha="center", va="top", fontsize=10, color=SUBTEXT)

fig.subplots_adjust(top=0.78, bottom=0.2, left=0.22, right=0.56)

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(BASE_DIR, f"forest_plot_vacina_ajustado_prob_resp.{ext}")
    fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
