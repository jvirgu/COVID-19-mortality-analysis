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
# Forest plot por número de doses vs. óbito hospitalar, com a regressão
# logística ajustada SEPARADAMENTE em cada ano (2021 e 2022 — 2020 fica de
# fora porque só existe a categoria "nenhuma dose" nesse ano, sem
# comparação possível). Apenas OR bruto: com o n menor por ano, um modelo
# ajustado com ~28 covariáveis não converge de forma confiável (mesma
# limitação já observada nas anotações por ano do script 17).
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "forest_doses_por_ano_pt.png")
OUTPUT_TIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "forest_doses_por_ano_pt.tiff")

DOSE_LABELS = {1: "1 dose", 2: "2 doses", 3: "3 doses", 4: "4 doses"}

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


def fit_or(formula, var, data):
    modelo = smf.glm(formula=formula, data=data,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    return np.exp(beta), np.exp(lo), np.exp(hi), p


# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS E AJUSTAR UM MODELO POR ANO (referência = nenhuma dose)
# ════════════════════════════════════════════════════════════
dados_full = pd.read_excel(XLSX_PATH)
dados_full["Ano"] = pd.to_datetime(dados_full["Data de Entrada"]).dt.year

n_total = len(dados_full)
n_deaths = int(dados_full["Óbito"].sum())

tabelas_por_ano = {}
for ano in [2021, 2022]:
    sub_ano = dados_full[dados_full["Ano"] == ano]
    doses_presentes = sorted(d for d in sub_ano["Vacinas"].unique() if d != 0)
    sub_dummies = pd.get_dummies(sub_ano, columns=["Vacinas"], dtype=int,
                                  drop_first=True)

    linhas = []
    for dose in doses_presentes:
        var = f"Vacinas_{dose}"
        n_geral = int((sub_ano["Vacinas"] == dose).sum())
        n_obito = int(sub_ano.loc[sub_ano["Vacinas"] == dose, "Óbito"].sum())
        try:
            OR, lo, hi, p = fit_or(f"Óbito ~ {var}", var, sub_dummies)
            if not (np.isfinite(OR) and OR > 1e-6):
                raise ValueError("não estimável")
        except Exception:
            OR = lo = hi = p = np.nan
        linhas.append({"dose": dose, "label": DOSE_LABELS[dose],
                        "n_geral": n_geral, "n_obito": n_obito,
                        "OR": OR, "IC_inf": lo, "IC_sup": hi, "p": p})
    tabelas_por_ano[ano] = pd.DataFrame(linhas)
    n_ano = len(sub_ano)
    obitos_ano = int(sub_ano["Óbito"].sum())
    tabelas_por_ano[f"{ano}_n"] = n_ano
    tabelas_por_ano[f"{ano}_obitos"] = obitos_ano

for ano in [2021, 2022]:
    print("=" * 70)
    print(f"Ano {ano} | n = {tabelas_por_ano[f'{ano}_n']} | "
          f"Óbitos = {tabelas_por_ano[f'{ano}_obitos']}")
    print(tabelas_por_ano[ano].to_string(index=False))
print("=" * 70)

# ════════════════════════════════════════════════════════════
# 2. FIGURA — DOIS FOREST PLOTS LADO A LADO (2021 | 2022)
# ════════════════════════════════════════════════════════════
n_rows_max = max(len(tabelas_por_ano[2021]), len(tabelas_por_ano[2022]))
fig_h = max(4.5, n_rows_max * 0.9 + 3.0)
fig, axes = plt.subplots(
    1, 5, figsize=(18, fig_h), facecolor=BG,
    gridspec_kw={"width_ratios": [2.2, 5, 1.6, 2.2, 5], "wspace": 0.05},
)
ax_lbl_2021, ax_2021, ax_gap, ax_lbl_2022, ax_2022 = axes
ax_gap.set_visible(False)


def desenha_ano(ax_lbl, ax_plot, df_raw, titulo, n_ano, obitos_ano, xlim):
    n_rows = len(df_raw)
    ax_lbl.set_facecolor(BG)
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(n_rows_max - 0.5, -0.5)
    for spine in ax_lbl.spines.values():
        spine.set_visible(False)
    ax_lbl.set_xticks([])
    ax_lbl.set_yticks([])

    for ax in (ax_lbl, ax_plot):
        for i in range(n_rows):
            bg_col = "#E8EDF2" if i % 2 == 0 else PANEL
            ax.axhspan(i - 0.42, i + 0.42, color=bg_col, alpha=0.55, zorder=0)

    for i, row in df_raw.iterrows():
        ax_lbl.text(0.98, i, f"{row['label']} (n={row['n_geral']})",
                    color=TEXT, fontsize=14.5, va="center", ha="right")

    ax_plot.set_facecolor(PANEL)
    ax_plot.set_xscale("log")
    ax_plot.xaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
    ax_plot.set_axisbelow(True)
    for spine in ax_plot.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax_plot.axvline(1.0, color=GOLD, linewidth=2.3, linestyle="--", zorder=2,
                     alpha=0.9)

    for i, row in df_raw.iterrows():
        OR, lo, hi, p = row["OR"], row["IC_inf"], row["IC_sup"], row["p"]
        if pd.isna(OR) or pd.isna(lo) or pd.isna(hi) or not np.isfinite(OR):
            ax_plot.text(0.5, i, "—", color=SUBTEXT, fontsize=13,
                         va="center", ha="center",
                         transform=ax_plot.get_yaxis_transform())
            continue
        cor = COR_PROT if OR < 1 else COR_RISCO
        sig = not np.isnan(p) and p < 0.05
        lo_plot = max(lo, xlim[0] * 1.02)
        hi_plot = min(hi, xlim[1] * 0.98) if np.isfinite(hi) else xlim[1] * 0.98
        ax_plot.plot([lo_plot, hi_plot], [i, i], color=cor, linewidth=3.5,
                     zorder=3, alpha=0.85, solid_capstyle="round")
        ax_plot.plot(OR, i, marker="D" if sig else "o",
                     markersize=8.5 if sig else 6.5, color=cor,
                     markerfacecolor=cor if sig else BG, markeredgecolor=cor,
                     markeredgewidth=1.5, zorder=5)
        hi_str = f"{min(hi, 999):.2f}" if np.isfinite(hi) else "∞"
        txt = f"{OR:.2f} ({lo:.2f}–{hi_str}){sig_stars(p)}"
        ax_plot.text(1.02, i, txt, transform=ax_plot.get_yaxis_transform(),
                     color=TEXT, fontsize=13.5, va="center", ha="left",
                     clip_on=False)

    ax_plot.set_yticks(range(n_rows))
    ax_plot.set_yticklabels([""] * n_rows)
    ax_plot.tick_params(axis="y", length=0)
    ax_plot.tick_params(axis="x", colors=SUBTEXT, labelsize=12.5)
    ax_plot.set_ylim(n_rows_max - 0.5, -0.5)
    ax_plot.set_xlim(*xlim)

    ticks = [t for t in [0.1, 0.3, 1, 3, 10, 30] if xlim[0] <= t <= xlim[1]]
    ax_plot.set_xticks(ticks)
    ax_plot.set_xticklabels([f"{t:g}".replace(".", ",") for t in ticks])
    ax_plot.xaxis.set_minor_locator(mticker.NullLocator())

    ax_plot.set_xlabel("Odds Ratio (escala log)", fontsize=16,
                        color=SUBTEXT, labelpad=6)
    ax_plot.set_title(f"{titulo}\nn = {n_ano} | Óbitos = {obitos_ano}",
                       fontsize=17, fontweight="bold", color=TEXT, pad=8)


desenha_ano(ax_lbl_2021, ax_2021, tabelas_por_ano[2021], "2021",
            tabelas_por_ano["2021_n"], tabelas_por_ano["2021_obitos"],
            xlim=(0.1, 30))
desenha_ano(ax_lbl_2022, ax_2022, tabelas_por_ano[2022], "2022",
            tabelas_por_ano["2022_n"], tabelas_por_ano["2022_obitos"],
            xlim=(0.1, 30))

# ── Legenda ───────────────────────────────────────────────────────────────
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

_escala = 4.5 / fig_h
_top = 1 - (1 - 0.74) * _escala
_y_titulo = 1 + (1.30 - 1) * _escala
_y_subtitulo = 1 + (1.17 - 1) * _escala
_y_divisor = 1 + (1.10 - 1) * _escala
_y_legenda = 1 + (1.02 - 1) * _escala
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=_top, wspace=0.05)

fig.text(0.50, _y_titulo,
         "Forest Plot — Número de Doses de Vacina vs. Óbito, por Ano",
         ha="center", va="top", fontsize=26, fontweight="bold", color=TEXT)
fig.text(0.50, _y_subtitulo,
         (f"Categoria de referência: nenhuma dose (dentro de cada ano) | "
          f"n total = {n_total} | Óbitos = {n_deaths} | "
          "regressão logística bruta ajustada separadamente por ano"),
         ha="center", va="top", fontsize=15.5, color=SUBTEXT)
fig.add_artist(plt.Line2D([0.10, 0.98], [_y_divisor, _y_divisor],
                           transform=fig.transFigure, color=BORDER,
                           linewidth=1.8))
fig.legend(handles=legend_elements, fontsize=11.5, frameon=False,
           labelcolor=TEXT, loc="upper center", ncol=5,
           bbox_to_anchor=(0.55, _y_legenda), columnspacing=1.5,
           handlelength=1.3, handletextpad=0.5)

plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
_buf_png = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
plt.savefig(_buf_png, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_png).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_png)

print(f"Gráfico salvo em: {OUTPUT_PNG} e {OUTPUT_TIFF}")
plt.show()
