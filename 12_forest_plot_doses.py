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
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "forest_doses_vs_obito.png")

DOSE_LABELS = {
    "Vacinas_1": "1 dose",
    "Vacinas_2": "2 doses",
    "Vacinas_3": "3 doses",
    "Vacinas_4": "4 doses",
}

# Covariates used to adjust the model (same set as the full logistic
# regression in 05_statistical_analysis_logisticregression.py, minus
# Prob_Infec, which is constant in this dataset and causes a
# rank-deficient design matrix).
ADJUST_VARS = [
    "Sexo", "Prob_Card", "CP", "Diabetes", "SRAG", "Choques", "Prob_neurol",
    "Prob_Hemat", "Cancer", "Prob_Resp", "Prob_Metab", "Prob_TGI",
    "Prob_Hep", "Prob_Hid_Elet", "Prob_AI_Infla", "Febre", "Outros",
    "Traumatismo", "COVID_CRÍTICA", "Prob_Renal", "LRA", "Dias_permanência",
    "Estado_Civil_1", "Estado_Civil_2", "Idade_cat_1", "Idade_cat_2",
    "Idade_cat_3", "Grau_Instrucao_1", "Grau_Instrucao_2",
]

# ════════════════════════════════════════════════════════════
# 1. LOAD DATA AND FIT MODELS (reference category = no dose)
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)
dados = pd.get_dummies(dados, columns=["Vacinas"], dtype=int, drop_first=True)
dose_vars = list(DOSE_LABELS.keys())

n_total = len(dados)
n_deaths = int(dados["Óbito"].sum())


def fit_or(formula, var, data):
    """Fits a binomial GLM and returns OR, 95% CI and p-value for `var`."""
    modelo = smf.glm(formula=formula, data=data,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    return np.exp(beta), np.exp(lo), np.exp(hi), p


rows = []
for var in dose_vars:
    n_ge = int(dados[var].sum())
    n_ob = int(dados.loc[dados[var] == 1, "Óbito"].sum())

    # Crude OR: dose dummy alone vs. Óbito
    OR, lo, hi, p = fit_or(f"Óbito ~ {var}", var, dados)

    # Adjusted OR: full model with all dose dummies + covariates
    formula_adj = ("Óbito ~ " + " + ".join(dose_vars) + " + "
                   + " + ".join(ADJUST_VARS))
    ORa, loa, hia, pa = fit_or(formula_adj, var, dados)

    rows.append({
        "label": DOSE_LABELS[var], "n_geral": n_ge, "n_obito": n_ob,
        "OR": OR, "IC_inf": lo, "IC_sup": hi, "p_OR": p,
        "ORa": ORa, "ICa_inf": loa, "ICa_sup": hia, "p_ORa": pa,
    })

df_raw = pd.DataFrame(rows)

print("=" * 70)
print(f"Table read from: {XLSX_PATH}")
print(f"Overall n = {n_total} | Deaths n = {n_deaths} | Reference = No dose")
print(df_raw.to_string())
print("=" * 70)

# ════════════════════════════════════════════════════════════
# 2. FOREST PLOT
# ════════════════════════════════════════════════════════════
# ── Palette ───────────────────────────────────────────────────────────────
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
GOLD = "#B08800"
COR_PROT = "#1D9E75"   # OR < 1 → protective
COR_RISCO = "#E07B39"  # OR > 1 → risk


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


# ── Figure: 4 columns — [labels] | [crude OR] | [gap] | [adjusted OR] ──────
n_rows = len(df_raw)
fig_h = max(6, n_rows * 0.9 + 3.0)
fig, axes = plt.subplots(
    1, 4, figsize=(16, fig_h), facecolor=BG,
    gridspec_kw={"width_ratios": [2.5, 5, 2.5, 5], "wspace": 0.04},
)
ax_labels, ax_or, ax_gap, ax_ora = axes

# Invisible spacer column
ax_gap.set_visible(False)

# ── Label panel (no axes) ───────────────────────────────────────────────────
ax_labels.set_facecolor(BG)
ax_labels.set_xlim(0, 1)
ax_labels.set_ylim(n_rows - 0.5, -0.5)
for spine in ax_labels.spines.values():
    spine.set_visible(False)
ax_labels.set_xticks([])
ax_labels.set_yticks([])

# Zebra stripes (synced across the 3 panels)
for ax in (ax_labels, ax_or, ax_ora):
    for i in range(n_rows):
        bg_col = "#E8EDF2" if i % 2 == 0 else PANEL
        ax.axhspan(i - 0.42, i + 0.42, color=bg_col, alpha=0.55, zorder=0)

# Labels in the left panel
for i, row in df_raw.iterrows():
    ax_labels.text(0.98, i, f"{row['label']} (n={row['n_geral']})",
                   color=TEXT, fontsize=16, va="center", ha="right")


# ── Generic OR panel function ───────────────────────────────────────────────
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
        OR = row[col_or]
        lo = row[col_lo]
        hi = row[col_hi]
        p = row[col_p]

        # Non-estimable OR (e.g. quasi-complete separation collapses OR to
        # ~0 with an unbounded CI)
        if (pd.isna(OR) or pd.isna(lo) or pd.isna(hi)
                or not np.isfinite(OR) or OR < 1e-6):
            ax.text(0.5, i, "—", color=SUBTEXT, fontsize=13,
                    va="center", ha="center",
                    transform=ax.get_yaxis_transform())
            continue

        cor = COR_PROT if OR < 1 else COR_RISCO
        sig = (p is not None) and not np.isnan(p) and (p < 0.05)

        # CI bar capped at the axis limits
        lo_plot = max(lo, xlim[0] * 1.02)
        hi_plot = min(hi, xlim[1] * 0.98) if np.isfinite(hi) else xlim[1] * 0.98
        ax.plot([lo_plot, hi_plot], [i, i],
                color=cor, linewidth=3.8, zorder=3, alpha=0.85,
                solid_capstyle="round")

        # Marker: diamond if significant, circle otherwise
        ax.plot(OR, i,
                marker="D" if sig else "o",
                markersize=9 if sig else 7,
                color=cor,
                markerfacecolor=cor if sig else BG,
                markeredgecolor=cor,
                markeredgewidth=1.6,
                zorder=5)

        # OR (CI) text + significance stars to the right of the panel
        hi_txt = min(hi, 9999) if np.isfinite(hi) else float("inf")
        hi_str = f"{hi_txt:.2f}" if np.isfinite(hi_txt) else "∞"
        txt = f"{OR:.2f} ({lo:.2f}–{hi_str}){sig_stars(p)}"
        ax.text(1.02, i, txt,
                transform=ax.get_yaxis_transform(),
                color=TEXT, fontsize=15, va="center", ha="left",
                clip_on=False)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([""] * n_rows)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=14)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Odds Ratio (log scale)", fontsize=20,
                  color=SUBTEXT, labelpad=6)
    ax.set_title(title, fontsize=20, fontweight="bold", color=TEXT, pad=5)


draw_panel(ax_or, "OR", "IC_inf", "IC_sup", "p_OR",
           "Crude OR (95% CI)", xlim=(0.05, 40))
draw_panel(ax_ora, "ORa", "ICa_inf", "ICa_sup", "p_ORa",
           "Adjusted OR (95% CI)", xlim=(0.05, 40))

# ── Legend ────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT,
                   label="Protective factor (OR < 1)"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO,
                   label="Risk factor (OR > 1)"),
    Line2D([0], [0], marker="D", color="none",
           markerfacecolor=TEXT, markeredgecolor=TEXT,
           markersize=5, label="Diamond = p < 0.05"),
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=BG, markeredgecolor=TEXT,
           markeredgewidth=1.2, markersize=6,
           label="Open circle = p ≥ 0.05"),
    Line2D([0], [0], color=GOLD, linewidth=1.2,
           linestyle="--", label="Reference line (OR = 1)"),
]
ax_or.legend(handles=legend_elements, fontsize=8, frameon=True,
             edgecolor=BORDER, facecolor=BG, labelcolor=TEXT,
             loc="lower left", framealpha=0.97,
             borderpad=0.9, handlelength=0.5)

# ── Main title ────────────────────────────────────────────────────────────
fig.text(0.50, 1.095,
         "Forest Plot — Vaccine Doses vs. In-Hospital Death",
         ha="center", va="top",
         fontsize=30, fontweight="bold", color=TEXT)

# ── Dynamic subtitle ────────────────────────────────────────────────────
subtitle = (f"Reference category: no dose | "
            f"Overall n = {n_total} | Deaths n = {n_deaths}")
fig.text(0.50, 1.015, subtitle,
         ha="center", va="top", fontsize=20, color=SUBTEXT)
fig.add_artist(plt.Line2D(
    [0.13, 0.97], [0.958, 0.958],
    transform=fig.transFigure, color=BORDER, linewidth=1.8))
fig.text(0.03, -0.101,
         "*** p<0.001 ** p<0.01 * p<0.05 | "
         "OR = Odds Ratio; CI = 95% Confidence Interval | "
         "Filled diamond = p < 0.05 | — = Unestimable OR | "
         "Adjusted for sex, comorbidities, age, marital status, "
         "education and length of stay",
         color=SUBTEXT, fontsize=12, style="italic")

plt.tight_layout(rect=[0, 0.018, 1, 1.970])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Chart saved to: {OUTPUT_PNG}")
plt.show()
