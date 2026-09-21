import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from PIL import Image
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Versão em português do forest plot por número de doses (12_forest_plot_
# doses.py), com o layout já usado no forest plot de status vacinal
# (14b): legenda em uma única linha, entre a referência/n e os painéis de
# OR, e eixo log sem sobreposição de rótulos. Exporta também em TIFF
# (comprimido com LZW).
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "forest_doses_vs_obito_pt.png")
OUTPUT_TIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "forest_doses_vs_obito_pt.tiff")

DOSE_LABELS = {1: "1 dose", 2: "2 doses", 3: "3 doses", 4: "4 doses"}

# Covariáveis usadas para ajuste (mesmo conjunto do forest plot original;
# Prob_Infec é excluída por ser constante no dataset, o que causa matriz
# de delineamento com posto deficiente).
ADJUST_VARS = [
    "Sexo", "Prob_Card", "CP", "Diabetes", "SRAG", "Choques", "Prob_neurol",
    "Prob_Hemat", "Cancer", "Prob_Resp", "Prob_Metab", "Prob_TGI",
    "Prob_Hep", "Prob_Hid_Elet", "Prob_AI_Infla", "Febre", "Outros",
    "Traumatismo", "COVID_CRÍTICA", "Prob_Renal", "LRA", "Dias_permanência",
    "Estado_Civil_1", "Estado_Civil_2", "Idade_cat_1", "Idade_cat_2",
    "Idade_cat_3", "Grau_Instrucao_1", "Grau_Instrucao_2",
]

# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS E AJUSTAR OS MODELOS (referência = nenhuma dose)
# ════════════════════════════════════════════════════════════
dados_todos = pd.read_excel(XLSX_PATH)
dados = pd.get_dummies(dados_todos, columns=["Vacinas"], dtype=int,
                        drop_first=True)

n_total = len(dados_todos)
n_deaths = int(dados_todos["Óbito"].sum())


def fit_or(formula, var, data):
    """Ajusta um GLM binomial e retorna OR, IC 95% e p-valor de `var`."""
    modelo = smf.glm(formula=formula, data=data,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    return np.exp(beta), np.exp(lo), np.exp(hi), p


linhas = []
for dose in [1, 2, 3, 4]:
    var = f"Vacinas_{dose}"
    n_geral = int(dados_todos.loc[dados_todos["Vacinas"] == dose].shape[0])
    n_obito = int(dados_todos.loc[dados_todos["Vacinas"] == dose, "Óbito"].sum())

    OR, lo, hi, p = fit_or(f"Óbito ~ {var}", var, dados)
    formula_adj = f"Óbito ~ {var} + " + " + ".join(ADJUST_VARS)
    ORa, loa, hia, pa = fit_or(formula_adj, var, dados)

    linhas.append({
        "label": DOSE_LABELS[dose], "n_geral": n_geral, "n_obito": n_obito,
        "OR": OR, "IC_inf": lo, "IC_sup": hi, "p_OR": p,
        "ORa": ORa, "ICa_inf": loa, "ICa_sup": hia, "p_ORa": pa,
    })

df_raw = pd.DataFrame(linhas)

print("=" * 70)
print(f"Tabela lida de: {XLSX_PATH}")
print(f"n total = {n_total} | Óbitos = {n_deaths} | Referência = nenhuma dose")
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
    ticks_principais = [t for t in [0.1, 0.3, 1, 3, 10, 30] if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks_principais)
    ax.set_xticklabels([f"{t:g}".replace(".", ",") for t in ticks_principais])
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel("Odds Ratio (escala log)", fontsize=20,
                  color=SUBTEXT, labelpad=6)
    ax.set_title(title, fontsize=20, fontweight="bold", color=TEXT, pad=5)


draw_panel(ax_or, "OR", "IC_inf", "IC_sup", "p_OR",
           "OR bruto (IC 95%)", xlim=(0.1, 30))
draw_panel(ax_ora, "ORa", "ICa_inf", "ICa_sup", "p_ORa",
           "OR ajustado (IC 95%)", xlim=(0.1, 30))

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
# Deixa espaço acima dos eixos para título, subtítulo e legenda. As frações
# de texto (fixas em polegadas) são escaladas pela altura da figura, que
# varia com o nº de linhas (doses).
_escala = 4.5 / fig_h
_top = 1 - (1 - 0.74) * _escala
_y_titulo = 1 + (1.30 - 1) * _escala
_y_subtitulo = 1 + (1.17 - 1) * _escala
_y_divisor = 1 + (1.10 - 1) * _escala
_y_legenda = 1 + (1.02 - 1) * _escala
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=_top, wspace=0.04)

# ── Título principal ────────────────────────────────────────────────────
fig.text(0.50, _y_titulo,
         "Forest Plot — Número de Doses de Vacina vs. Óbito Hospitalar",
         ha="center", va="top",
         fontsize=28, fontweight="bold", color=TEXT)

# ── Subtítulo dinâmico ───────────────────────────────────────────────────
subtitle = (f"Categoria de referência: nenhuma dose | "
            f"n total = {n_total} | Óbitos = {n_deaths}")
fig.text(0.50, _y_subtitulo, subtitle,
         ha="center", va="top", fontsize=18, color=SUBTEXT)
fig.add_artist(plt.Line2D(
    [0.13, 0.97], [_y_divisor, _y_divisor],
    transform=fig.transFigure, color=BORDER, linewidth=1.8))

# ── Legenda (uma única linha, entre a referência e os painéis de OR) ──────
fig.legend(handles=legend_elements, fontsize=12.5, frameon=False,
           labelcolor=TEXT, loc="upper center", ncol=5,
           bbox_to_anchor=(0.55, _y_legenda), columnspacing=1.6,
           handlelength=1.4, handletextpad=0.6)

plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)

# TIFF com compressão LZW (o TIFF sem compressão do matplotlib passa de 50MB)
_buf_png = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
plt.savefig(_buf_png, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_png).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_png)

print(f"Gráfico salvo em: {OUTPUT_PNG} e {OUTPUT_TIFF}")
plt.show()
