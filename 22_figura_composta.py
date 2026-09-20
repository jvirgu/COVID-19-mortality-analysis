import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from PIL import Image
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Figura composta:
#   Painel A — Kaplan-Meier (vacinação) por data de admissão e por tempo
#              de internação (2021-2022)
#   Painel B — Forest plot: status vacinal vs. óbito hospitalar
#              (OR bruto e ajustado)
#   Painel C — Tendências temporais de internações e óbitos por onda
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = "figura_composta_vacinacao.png"
OUTPUT_TIFF = "figura_composta_vacinacao.tiff"

# ── Paleta ────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
GOLD = "#B08800"
COR_PROT = "#1D9E75"
COR_RISCO = "#E07B39"
COR_NAO_VAC = "#E0703A"
COR_VAC = "#1D9E75"

ADJUST_VARS = [
    "Sexo", "Prob_Card", "CP", "Diabetes", "SRAG", "Choques", "Prob_neurol",
    "Prob_Hemat", "Cancer", "Prob_Resp", "Prob_Metab", "Prob_TGI",
    "Prob_Hep", "Prob_Hid_Elet", "Prob_AI_Infla", "Febre", "Outros",
    "Traumatismo", "COVID_CRÍTICA", "Prob_Renal", "LRA", "Dias_permanência",
    "Estado_Civil_1", "Estado_Civil_2", "Idade_cat_1", "Idade_cat_2",
    "Idade_cat_3", "Grau_Instrucao_1", "Grau_Instrucao_2",
]


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


# ════════════════════════════════════════════════════════════
# CARREGAR DADOS
# ════════════════════════════════════════════════════════════
dados_full = pd.read_excel(XLSX_PATH)
dados_full["Data de Entrada"] = pd.to_datetime(dados_full["Data de Entrada"])
dados_full["Ano"] = dados_full["Data de Entrada"].dt.year

# Subconjunto 2021-2022 (período com vacinação) para o painel A
dados_km = dados_full[dados_full["Ano"].isin([2021, 2022])].copy()
data_min = dados_km["Data de Entrada"].min()
dados_km["Dias_desde_inicio"] = (dados_km["Data de Entrada"] - data_min).dt.days
dados_km["Grupo"] = dados_km["Vacinado"].map({0: "Não vacinados", 1: "Vacinados"})
CORES_KM = {"Não vacinados": COR_NAO_VAC, "Vacinados": COR_VAC}

# ════════════════════════════════════════════════════════════
# FIGURA — GRIDSPEC 2 LINHAS x 4 COLUNAS
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 17), facecolor=BG)
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.85], hspace=0.42, wspace=0.55,
                       left=0.045, right=0.985, top=0.90, bottom=0.10)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_a2 = fig.add_subplot(gs[0, 1])
ax_b1 = fig.add_subplot(gs[0, 2])
ax_b2 = fig.add_subplot(gs[0, 3])
ax_c = fig.add_subplot(gs[1, :])


# ════════════════════════════════════════════════════════════
# PAINEL A — KAPLAN-MEIER (VACINAÇÃO)
# ════════════════════════════════════════════════════════════
def plota_km(ax, coluna_tempo, titulo_eixo_x, eixo_como_data=False):
    for grupo in ["Não vacinados", "Vacinados"]:
        sub = dados_km[dados_km["Grupo"] == grupo]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[coluna_tempo], event_observed=sub["Óbito"],
                label=f"{grupo} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color=CORES_KM[grupo], linewidth=2.0,
                                    ci_alpha=0.13, show_censors=False)

    ax.set_xlabel(titulo_eixo_x, fontsize=11, color=SUBTEXT)
    ax.set_ylabel("Probabilidade de sobrevida", fontsize=11, color=SUBTEXT)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", labelsize=9.5, colors=SUBTEXT)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if eixo_como_data:
        def fmt_data(x, _):
            return (data_min + pd.Timedelta(days=x)).strftime("%b/%Y")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_data))
        ax.tick_params(axis="x", rotation=30)


def teste_logrank(coluna_tempo):
    a = dados_km[dados_km["Grupo"] == "Não vacinados"]
    b = dados_km[dados_km["Grupo"] == "Vacinados"]
    resultado = logrank_test(a[coluna_tempo], b[coluna_tempo],
                              event_observed_A=a["Óbito"],
                              event_observed_B=b["Óbito"])
    return resultado.p_value


plota_km(ax_a1, "Dias_desde_inicio", "Data de admissão", eixo_como_data=True)
p_lr_data = teste_logrank("Dias_desde_inicio")
ax_a1.text(0.97, 0.97, f"Log-rank\n{formata_p(p_lr_data)}",
           transform=ax_a1.transAxes, ha="right", va="top",
           fontsize=9, style="italic", color=TEXT)
ax_a1.set_title("Por data de admissão", fontsize=12.5, fontweight="bold",
                color=TEXT, pad=8)

plota_km(ax_a2, "Dias_permanência", "Tempo de internação (dias)")
p_lr_dias = teste_logrank("Dias_permanência")
ax_a2.text(0.97, 0.97, f"Log-rank\n{formata_p(p_lr_dias)}",
           transform=ax_a2.transAxes, ha="right", va="top",
           fontsize=9, style="italic", color=TEXT)
ax_a2.set_title("Por tempo de internação", fontsize=12.5, fontweight="bold",
                color=TEXT, pad=8)

fig.text(gs[0, 0].get_position(fig).x0 - 0.028,
         gs[0, 0].get_position(fig).y1 + 0.025,
         "A", fontsize=26, fontweight="bold", color=TEXT)
fig.text((gs[0, 0].get_position(fig).x0 + gs[0, 1].get_position(fig).x1) / 2,
         gs[0, 0].get_position(fig).y1 + 0.045,
         "Sobrevida hospitalar por status vacinal (2021-2022)",
         ha="center", fontsize=13.5, fontweight="bold", color=TEXT)

# ════════════════════════════════════════════════════════════
# PAINEL B — FOREST PLOT: STATUS VACINAL vs. ÓBITO
# ════════════════════════════════════════════════════════════
n_total = len(dados_full)
n_deaths = int(dados_full["Óbito"].sum())
n_vacinado = int(dados_full["Vacinado"].sum())
n_vacinado_obito = int(dados_full.loc[dados_full["Vacinado"] == 1, "Óbito"].sum())


def fit_or(formula, var, data):
    modelo = smf.glm(formula=formula, data=data,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    return np.exp(beta), np.exp(lo), np.exp(hi), p


OR, lo, hi, p = fit_or("Óbito ~ Vacinado", "Vacinado", dados_full)
formula_adj = "Óbito ~ Vacinado + " + " + ".join(ADJUST_VARS)
ORa, loa, hia, pa = fit_or(formula_adj, "Vacinado", dados_full)

df_raw = pd.DataFrame([{
    "label": "Vacinados", "n_geral": n_vacinado, "OR": OR, "IC_inf": lo,
    "IC_sup": hi, "p_OR": p, "ORa": ORa, "ICa_inf": loa, "ICa_sup": hia,
    "p_ORa": pa,
}])


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


def draw_forest_panel(ax, col_or, col_lo, col_hi, col_p, titulo, xlim):
    n_rows = len(df_raw)
    ax.set_facecolor(PANEL)
    ax.set_xscale("log")
    ax.xaxis.grid(True, color=BORDER, linewidth=0.8, zorder=0, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.axvline(1.0, color=GOLD, linewidth=2.0, linestyle="--", zorder=2, alpha=0.9)
    ax.axhspan(-0.42, 0.42, color="#E8EDF2", alpha=0.55, zorder=0)

    row = df_raw.iloc[0]
    OR_, lo_, hi_, p_ = row[col_or], row[col_lo], row[col_hi], row[col_p]
    cor = COR_PROT if OR_ < 1 else COR_RISCO
    sig = (p_ is not None) and not np.isnan(p_) and (p_ < 0.05)

    lo_plot = max(lo_, xlim[0] * 1.02)
    hi_plot = min(hi_, xlim[1] * 0.98) if np.isfinite(hi_) else xlim[1] * 0.98
    ax.plot([lo_plot, hi_plot], [0, 0], color=cor, linewidth=3.2, zorder=3,
            alpha=0.85, solid_capstyle="round")
    ax.plot(OR_, 0, marker="D" if sig else "o", markersize=8 if sig else 6.5,
            color=cor, markerfacecolor=cor if sig else BG,
            markeredgecolor=cor, markeredgewidth=1.5, zorder=5)

    txt = f"{OR_:.2f} ({lo_:.2f}–{hi_:.2f}){sig_stars(p_)}"
    ax.text(OR_, -0.55, txt, color=TEXT, fontsize=10.5, va="top", ha="center")

    ax.set_yticks([0])
    ax.set_yticklabels([f"Vacinados\n(n={n_vacinado})"], fontsize=10.5, color=TEXT)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=9.5)
    ax.set_ylim(1.0, -1.0)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Odds Ratio (escala log)", fontsize=10.5, color=SUBTEXT, labelpad=5)
    ax.set_title(titulo, fontsize=12.5, fontweight="bold", color=TEXT, pad=8)


draw_forest_panel(ax_b1, "OR", "IC_inf", "IC_sup", "p_OR",
                   "OR bruto (IC 95%)", xlim=(0.3, 3))
draw_forest_panel(ax_b2, "ORa", "ICa_inf", "ICa_sup", "p_ORa",
                   "OR ajustado (IC 95%)", xlim=(0.3, 3))

fig.text(gs[0, 2].get_position(fig).x0 - 0.028,
         gs[0, 2].get_position(fig).y1 + 0.025,
         "B", fontsize=26, fontweight="bold", color=TEXT)
fig.text((gs[0, 2].get_position(fig).x0 + gs[0, 3].get_position(fig).x1) / 2,
         gs[0, 2].get_position(fig).y1 + 0.045,
         "Status vacinal vs. óbito hospitalar", ha="center",
         fontsize=13.5, fontweight="bold", color=TEXT)

# ════════════════════════════════════════════════════════════
# PAINEL C — TENDÊNCIAS TEMPORAIS POR ONDA
# ════════════════════════════════════════════════════════════
mensal = dados_full.groupby(dados_full["Data de Entrada"].dt.to_period("M")).agg(
    n_internacoes=("Data de Entrada", "count"),
    n_obitos=("Óbito", "sum")
).reset_index()
mensal["n_altas"] = mensal["n_internacoes"] - mensal["n_obitos"]
mensal.rename(columns={"Data de Entrada": "mes"}, inplace=True)
mensal["mes_dt"] = mensal["mes"].dt.to_timestamp()

ondas = [
    {"label": "1ª Onda", "inicio": "2020-01-01", "fim": "2020-09-30", "cor": "#7b5ea7"},
    {"label": "2ª Onda", "inicio": "2020-10-01", "fim": "2021-05-31", "cor": "#c0823a"},
    {"label": "3ª Onda", "inicio": "2021-07-01", "fim": "2021-11-30", "cor": "#b03060"},
    {"label": "4ª Onda", "inicio": "2021-12-01", "fim": "2022-12-30", "cor": "#5b8db8"},
]

for onda in ondas:
    ax_c.axvspan(pd.to_datetime(onda["inicio"]), pd.to_datetime(onda["fim"]),
                 alpha=0.3, color=onda["cor"])

ax_c.fill_between(mensal["mes_dt"], mensal["n_internacoes"], alpha=0.6, color="#2980b9")
ax_c.plot(mensal["mes_dt"], mensal["n_internacoes"], color="#2980b9", linewidth=2.4)
ax_c.fill_between(mensal["mes_dt"], mensal["n_altas"], alpha=0.6, color="#27ae60")
ax_c.plot(mensal["mes_dt"], mensal["n_altas"], color="#1e8449", linewidth=2.4)
ax_c.fill_between(mensal["mes_dt"], mensal["n_obitos"], alpha=0.7, color="#e74c3c")
ax_c.plot(mensal["mes_dt"], mensal["n_obitos"], color="#c0392b", linewidth=2.4)

vacinacao = pd.to_datetime("2021-01-19")
ax_c.axvline(x=vacinacao, color="black", linestyle="--", linewidth=2.4)
ax_c.text(vacinacao + pd.Timedelta(days=5), 53,
          "Início da Vacinação\n(19/01/2021)", fontsize=10.5, color="black", va="top")

ax_c.set_xlim(pd.to_datetime("2020-01-01"), pd.to_datetime("2022-12-31"))
ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
ax_c.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax_c.tick_params(axis="x", rotation=40, labelsize=10.5, colors=SUBTEXT)
ax_c.tick_params(axis="y", labelsize=10.5, colors=SUBTEXT)
ax_c.set_xlabel("Período", fontsize=12.5, fontweight="bold", color=TEXT)
ax_c.set_ylabel("Número de Pacientes", fontsize=12.5, fontweight="bold", color=TEXT)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.set_ylim(0, 58)

legend_elementos_c = [
    Line2D([0], [0], color="#2980b9", lw=3, label="Internações (total)"),
    Line2D([0], [0], color="#27ae60", lw=3, label="Alta"),
    Line2D([0], [0], color="#e74c3c", lw=3, label="Óbito"),
    Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="Início da Vacinação"),
    plt.Rectangle((0, 0), 1, 1, fc="#7b5ea7", alpha=0.5, label="1ª Onda"),
    plt.Rectangle((0, 0), 1, 1, fc="#c0823a", alpha=0.5, label="2ª Onda"),
    plt.Rectangle((0, 0), 1, 1, fc="#b03060", alpha=0.5, label="3ª Onda"),
    plt.Rectangle((0, 0), 1, 1, fc="#5b8db8", alpha=0.5, label="4ª Onda"),
]
ax_c.legend(handles=legend_elementos_c, loc="upper right", fontsize=9,
            framealpha=0.92, ncol=2)

stats_ondas = [
    {"x": "2020-05-01", "label": "1ª Onda\n(ref.)"},
    {"x": "2020-11-15", "label": "2ª Onda\nOR=0,67 [0,45–0,98]\np=0,041"},
    {"x": "2021-09-01", "label": "3ª Onda\nOR=0,64 [0,40–1,03]\np=0,068"},
    {"x": "2022-04-01", "label": "4ª Onda\nOR=0,14 [0,09–0,23]\np<0,001"},
]
for s in stats_ondas:
    ax_c.text(pd.to_datetime(s["x"]), 0.83, s["label"],
              transform=ax_c.get_xaxis_transform(), fontsize=8, color="#333333",
              ha="center", va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                        alpha=0.75, linewidth=0.8))

ax_c.text(0.995, 0.985, "χ²=72,88, gl=3, p<0,001",
          transform=ax_c.transAxes, fontsize=9.5, color="#555555",
          ha="right", va="top",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                    alpha=0.95, linewidth=0.6))

pos_c = ax_c.get_position(fig)
fig.text(pos_c.x0 - 0.028, pos_c.y1 + 0.028,
         "C", fontsize=26, fontweight="bold", color=TEXT)
fig.text((pos_c.x0 + pos_c.x1) / 2, pos_c.y1 + 0.028,
         "Tendências temporais de internações e óbitos por onda de COVID-19",
         ha="center", fontsize=13.5, fontweight="bold", color=TEXT)

# ════════════════════════════════════════════════════════════
# TÍTULO GERAL E RODAPÉ
# ════════════════════════════════════════════════════════════
fig.suptitle("Vacinação contra COVID-19: Sobrevida, Associação com Óbito e "
             "Tendências Temporais", fontsize=20, fontweight="bold", color=TEXT,
             y=0.975)
fig.text(0.5, 0.945,
          f"n total = {n_total} | Óbitos = {n_deaths} | "
          f"Não vacinados = {n_total - n_vacinado} | Vacinados = {n_vacinado}",
          ha="center", fontsize=12, color=SUBTEXT)
fig.text(0.045, 0.015,
          "OR = Odds Ratio; IC = Intervalo de Confiança de 95% | Laranja = "
          "OR > 1 (risco), verde = OR < 1 (protetor), linha dourada = OR = 1 | "
          "Painel B ajustado por sexo, comorbidades, idade, estado civil, "
          "escolaridade e tempo de internação | Evento nas curvas de "
          "Kaplan-Meier = óbito (alta = censura)",
          ha="left", va="bottom", fontsize=9.5, style="italic", color=SUBTEXT)

fig.savefig(OUTPUT_PNG, dpi=200, facecolor=BG, bbox_inches="tight")
Image.open(OUTPUT_PNG).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
print(f"Figura salva em: {OUTPUT_PNG} e {OUTPUT_TIFF}")
plt.show()
