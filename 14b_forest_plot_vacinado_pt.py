import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "forest_vacinado_vs_obito_pt.png")

# Covariates used to adjust the model (same set used in the dose forest
# plot; Prob_Infec is excluded because it is constant in this dataset and
# causes a rank-deficient design matrix).
ADJUST_VARS = [
    "Sexo", "Prob_Card", "CP", "Diabetes", "SRAG", "Choques", "Prob_neurol",
    "Prob_Hemat", "Cancer", "Prob_Resp", "Prob_Metab", "Prob_TGI",
    "Prob_Hep", "Prob_Hid_Elet", "Prob_AI_Infla", "Febre", "Outros",
    "Traumatismo", "COVID_CRÍTICA", "Prob_Renal", "LRA", "Dias_permanência",
    "Estado_Civil_1", "Estado_Civil_2", "Idade_cat_1", "Idade_cat_2",
    "Idade_cat_3", "Grau_Instrucao_1", "Grau_Instrucao_2",
]

# ════════════════════════════════════════════════════════════
# 1. LOAD DATA AND FIT MODELS (reference category = not vaccinated)
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)

n_total = len(dados)
n_deaths = int(dados["Óbito"].sum())
n_vacinado = int(dados["Vacinado"].sum())
n_vacinado_obito = int(dados.loc[dados["Vacinado"] == 1, "Óbito"].sum())


def fit_or(formula, var, data):
    """Fits a binomial GLM and returns OR, 95% CI and p-value for `var`."""
    modelo = smf.glm(formula=formula, data=data,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    return np.exp(beta), np.exp(lo), np.exp(hi), p

# Crude OR: Vacinado alone vs. Óbito
OR, lo, hi, p = fit_or("Óbito ~ Vacinado", "Vacinado", dados)

# Adjusted OR: Vacinado + covariates
formula_adj = "Óbito ~ Vacinado + " + " + ".join(ADJUST_VARS)
ORa, loa, hia, pa = fit_or(formula_adj, "Vacinado", dados)

df_raw = pd.DataFrame([{
    "label": "Vacinados", "n_geral": n_vacinado, "n_obito": n_vacinado_obito,
    "OR": OR, "IC_inf": lo, "IC_sup": hi, "p_OR": p,
    "ORa": ORa, "ICa_inf": loa, "ICa_sup": hia, "p_ORa": pa,
}])

print("=" * 70)
print(f"Tabela lida de: {XLSX_PATH}")
print(f"n total = {n_total} | Óbitos = {n_deaths} | Referência = não vacinado")
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
fig_h = max(4.5, n_rows * 0.9 + 3.0)
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

        # Non-estimable OR (e.g. quasi-complete separation)
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

    # Ticks explícitos no eixo log — evita a sobreposição dos rótulos
    # automáticos (3×10⁻¹, 4×10⁻¹, 6×10⁻¹, ...) do LogLocator padrão.
    ticks_principais = [t for t in [0.3, 0.5, 1, 2, 3] if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks_principais)
    ax.set_xticklabels([f"{t:g}".replace(".", ",") for t in ticks_principais])
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel("Odds Ratio (escala log)", fontsize=20,
                  color=SUBTEXT, labelpad=6)
    ax.set_title(title, fontsize=20, fontweight="bold", color=TEXT, pad=5)


draw_panel(ax_or, "OR", "IC_inf", "IC_sup", "p_OR",
           "OR bruto (IC 95%)", xlim=(0.3, 3))
draw_panel(ax_ora, "ORa", "ICa_inf", "ICa_sup", "p_ORa",
           "OR ajustado (IC 95%)", xlim=(0.3, 3))

# ── Legenda ───────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT,
                   label="Fator protetor (OR < 1)"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO,
                   label="Fator de risco (OR > 1)"),
    Line2D([0], [0], marker="D", color="none",
           markerfacecolor=TEXT, markeredgecolor=TEXT,
           markersize=5, label="Losango = p < 0,05"),
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=BG, markeredgecolor=TEXT,
           markeredgewidth=1.2, markersize=6,
           label="Círculo aberto = p ≥ 0,05"),
    Line2D([0], [0], color=GOLD, linewidth=1.2,
           linestyle="--", label="Linha de referência (OR = 1)"),
]
# Deixa espaço acima dos eixos para título, subtítulo e legenda
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.74, wspace=0.04)

# ── Título principal ────────────────────────────────────────────────────
fig.text(0.50, 1.30,
         "Forest Plot — Status Vacinal vs. Óbito Hospitalar",
         ha="center", va="top",
         fontsize=30, fontweight="bold", color=TEXT)

# ── Subtítulo dinâmico ───────────────────────────────────────────────────
subtitle = (f"Categoria de referência: não vacinado | "
            f"n total = {n_total} | Óbitos = {n_deaths}")
fig.text(0.50, 1.17, subtitle,
         ha="center", va="top", fontsize=20, color=SUBTEXT)
fig.add_artist(plt.Line2D(
    [0.13, 0.97], [1.10, 1.10],
    transform=fig.transFigure, color=BORDER, linewidth=1.8))

# ── Legenda (uma única linha, entre a referência e os painéis de OR) ──────
fig.legend(handles=legend_elements, fontsize=13, frameon=False,
           labelcolor=TEXT, loc="upper center", ncol=5,
           bbox_to_anchor=(0.55, 1.02), columnspacing=1.6,
           handlelength=1.4, handletextpad=0.6)
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
plt.show()
