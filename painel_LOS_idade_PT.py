"""
Painel A) Trend in Length of Stay by Age
Painel B) Length of Stay by Age Group
Fonte: New_pacientes703.xlsx (coorte tratada, n=703)

Analise estatistica: teste de normalidade de Shapiro-Wilk (para justificar a
abordagem nao parametrica) + teste de Mann-Whitney U (comparacao Death vs
Discharge). As legendas estatisticas NAO sao desenhadas como caixa no rodape
da figura; os resultados completos sao impressos no console para uso em
texto/nota.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "New_pacientes703.xlsx")
OUT_DIR = BASE_DIR

# ----------------------------------------------------------------------
# Fonte (Arial nao disponivel neste ambiente -> Liberation Sans,
# metric-compativel com Arial; fallback portatil)
# ----------------------------------------------------------------------
def _registrar_arial():
    candidatos = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    ]
    for c in candidatos:
        if os.path.exists(c):
            fm.fontManager.addfont(c)
            nome = fm.FontProperties(fname=c).get_name()
            plt.rcParams["font.family"] = nome
            return nome
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


_registrar_arial()

VERDE = "#1D9E75"   # Discharge (alta)
LARANJA = "#E07B39"  # Death (obito)

plt.rcParams.update({
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})

# ----------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
df["Obito_lbl"] = df["Óbito"].map({0: "Alta", 1: "Óbito"})
LABELS_IDADE_CAT = {0: "0-18", 1: "19-40", 2: "41-60", 3: "+60"}
df["Grupo_etario"] = df["Idade_cat"].map(LABELS_IDADE_CAT)
ordem_grupos = ["0-18", "19-40", "41-60", "+60"]

# ----------------------------------------------------------------------
# Tamanhos amostrais (n) por idade, grupo etario e desfecho
# ----------------------------------------------------------------------
n_total = len(df)
n_disch_total = int((df["Óbito"] == 0).sum())
n_death_total = int((df["Óbito"] == 1).sum())

n_por_grupo = df.groupby("Grupo_etario").size().reindex(ordem_grupos)
n_por_grupo_desfecho = (
    df.groupby(["Grupo_etario", "Obito_lbl"]).size().unstack().reindex(ordem_grupos)
)
n_por_idade = df.groupby(["Idade", "Obito_lbl"])["Dias_permanência"].size().unstack()

# ========================================================================
# ANALISE ESTATISTICA: Shapiro-Wilk (normalidade) + Mann-Whitney U
# ========================================================================

def estrelas(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

# --- Shapiro-Wilk por desfecho (geral) ---
sw_disch = stats.shapiro(df[df["Óbito"] == 0]["Dias_permanência"])
sw_death = stats.shapiro(df[df["Óbito"] == 1]["Dias_permanência"])

# --- Mann-Whitney U global (Discharge vs Death, referente ao Painel A) ---
mw_global = stats.mannwhitneyu(
    df[df["Óbito"] == 1]["Dias_permanência"],
    df[df["Óbito"] == 0]["Dias_permanência"],
    alternative="two-sided",
)

# --- Mann-Whitney U por grupo etario (Painel B) + Shapiro-Wilk por celula ---
mw_resultados = {}
sw_por_grupo = {}
for g in ordem_grupos:
    sub = df[df["Grupo_etario"] == g]
    death = sub[sub["Óbito"] == 1]["Dias_permanência"]
    disch = sub[sub["Óbito"] == 0]["Dias_permanência"]
    u = stats.mannwhitneyu(death, disch, alternative="two-sided")
    mw_resultados[g] = dict(
        p=u.pvalue, med_death=death.median(), med_disch=disch.median(),
        n_death=len(death), n_disch=len(disch)
    )
    sw_por_grupo[g] = {}
    for lbl, serie in [("Alta", disch), ("Óbito", death)]:
        sw_por_grupo[g][lbl] = stats.shapiro(serie) if len(serie) >= 3 else None

# --- impressao no console (para copiar como texto/nota) ---
print("=" * 70)
print("SHAPIRO-WILK (normalidade da distribuicao do LOS, por desfecho)")
print("=" * 70)
print(f"Discharge (n={n_disch_total}): W = {sw_disch.statistic:.4f}, "
      f"p {'< 0.001' if sw_disch.pvalue < 0.001 else '= ' + format(sw_disch.pvalue, '.4f')}")
print(f"Death     (n={n_death_total}): W = {sw_death.statistic:.4f}, "
      f"p {'< 0.001' if sw_death.pvalue < 0.001 else '= ' + format(sw_death.pvalue, '.4f')}")
print()
print("=" * 70)
print("MANN-WHITNEY U (Death vs Discharge)")
print("=" * 70)
p_glob = mw_global.pvalue
print(f"Global (todas as idades, Painel A): U = {mw_global.statistic:.1f}, "
      f"p {'< 0.001' if p_glob < 0.001 else '= ' + format(p_glob, '.4f')} ({estrelas(p_glob)})")
for g in ordem_grupos:
    r = mw_resultados[g]
    txt_p = "< 0.001" if r["p"] < 0.001 else f"= {r['p']:.4f}"
    print(f"{g}: Death md={r['med_death']} (n={r['n_death']}) vs Discharge md={r['med_disch']} (n={r['n_disch']}) "
          f"-> p {txt_p} ({estrelas(r['p'])})")
print()

# ========================================================================
# FIGURA (sem caixas de texto estatistico no rodape)
# ========================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.6))
fig.subplots_adjust(wspace=0.34, top=0.85, bottom=0.14, left=0.055, right=0.87)

# ---------------------------- PAINEL A ----------------------------------
media_idade = (
    df.groupby(["Idade", "Obito_lbl"])["Dias_permanência"]
    .mean()
    .unstack()
    .reindex(range(0, df["Idade"].max() + 1))
)

for col, cor, z in [("Alta", VERDE, 3), ("Óbito", LARANJA, 2)]:
    y = media_idade[col].values
    x = media_idade.index.values
    n_col = n_disch_total if col == "Alta" else n_death_total
    pct_col = 100 * n_col / n_total
    axA.plot(x, y, color=cor, linewidth=1.4, label=f"{col} (n = {n_col}, {pct_col:.1f}%)", zorder=z)
    axA.fill_between(x, y, 0, color=cor, alpha=0.18, zorder=1,
                      where=~np.isnan(y.astype(float)), interpolate=False)

axA.set_title("Tendência do Tempo de Internação por Idade", fontsize=14, fontweight="bold", color="#333333", pad=10)
axA.set_xlabel("Idade", fontsize=11)
axA.set_ylabel("Tempo de internação (dias)", fontsize=11)
axA.set_xlim(0, df["Idade"].max())
axA.set_ylim(0, None)
axA.grid(axis="both", alpha=0.5)
axA.set_axisbelow(True)

# marcadores de n por idade exata (a cada 5 anos, para nao poluir)
for idade_marca in range(0, df["Idade"].max() + 1, 5):
    if idade_marca not in n_por_idade.index:
        continue
    for col, cor, dy in [("Alta", VERDE, 6), ("Óbito", LARANJA, -6)]:
        if col not in n_por_idade.columns:
            continue
        n_val = n_por_idade.loc[idade_marca, col]
        if pd.isna(n_val):
            continue
        y_pt = media_idade.loc[idade_marca, col]
        if pd.isna(y_pt):
            continue
        axA.annotate(f"{int(n_val)}", xy=(idade_marca, y_pt), xytext=(0, dy),
                     textcoords="offset points", ha="center", fontsize=5.6, color=cor, alpha=0.85)

axA.legend(title="Desfecho", loc="upper right", frameon=True, framealpha=0.9, fontsize=9, title_fontsize=9.5)
p_glob_txt = "p < 0,001" if p_glob < 0.001 else f"p = {p_glob:.3f}".replace(".", ",")
sig_glob = f" ({estrelas(p_glob)})" if p_glob < 0.05 else ""
axA.text(0.012, 0.98, f"Mann-Whitney U (global, todas as idades): {p_glob_txt}{sig_glob}",
         transform=axA.transAxes, fontsize=8.5, va="top", ha="left", color="#333333",
         bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#cccccc", lw=0.7))
axA.text(-0.09, 1.05, "A)", transform=axA.transAxes, fontsize=15, fontweight="bold")

# ---------------------------- PAINEL B ----------------------------------
sns.boxplot(
    data=df, x="Grupo_etario", y="Dias_permanência", hue="Obito_lbl",
    order=ordem_grupos, hue_order=["Alta", "Óbito"],
    palette={"Alta": VERDE, "Óbito": LARANJA},
    ax=axB, showfliers=False, width=0.6, linewidth=1.0,
    boxprops=dict(alpha=0.9), medianprops=dict(color="#222222", linewidth=1.3)
)
sns.stripplot(
    data=df, x="Grupo_etario", y="Dias_permanência", hue="Obito_lbl",
    order=ordem_grupos, hue_order=["Alta", "Óbito"],
    dodge=True, ax=axB, palette={"Alta": "#4d4d4d", "Óbito": "#4d4d4d"},
    alpha=0.35, size=3, jitter=0.18, legend=False
)

axB.set_title("Tempo de Internação por Grupo Etário", fontsize=14, fontweight="bold", color="#333333", pad=10)
axB.set_xlabel("Grupo Etário", fontsize=11)
axB.set_ylabel("Tempo de internação (dias)", fontsize=11)
axB.grid(axis="y", alpha=0.5)
axB.set_axisbelow(True)

# rotulos do eixo x com o N total de cada grupo etario (absoluto e %)
novos_ticks = [f"{g}\n(N = {int(n_por_grupo[g])}, {100*n_por_grupo[g]/n_total:.1f}%)" for g in ordem_grupos]
axB.set_xticks(range(len(ordem_grupos)))
axB.set_xticklabels(novos_ticks, fontsize=9.5)

handles, labels_ = axB.get_legend_handles_labels()
labels_n = [f"Alta (n = {n_disch_total}, {100*n_disch_total/n_total:.1f}%)",
            f"Óbito (n = {n_death_total}, {100*n_death_total/n_total:.1f}%)"]
axB.legend(handles[:2], labels_n, title="Desfecho", loc="upper left", bbox_to_anchor=(1.01, 1.02),
           frameon=True, framealpha=0.9, fontsize=9, title_fontsize=9.5, borderaxespad=0)

# n de cada caixa (idade_cat x desfecho) - absoluto e % dentro do grupo etario,
# posicionado no topo do bigode de cada caixa
def _whisker_top(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    lim = q3 + 1.5 * iqr
    dentro = serie[serie <= lim]
    return dentro.max() if len(dentro) else serie.max()

for i, g in enumerate(ordem_grupos):
    n_grupo = int(n_por_grupo[g])
    for dx, dy, col, cor in [(-0.20, 4, "Alta", "#0f6b4c"), (0.20, 12, "Óbito", "#a85320")]:
        sub = df[(df["Grupo_etario"] == g) & (df["Obito_lbl"] == col)]["Dias_permanência"]
        n_val = int(n_por_grupo_desfecho.loc[g, col])
        pct_val = 100 * n_val / n_grupo
        wt = _whisker_top(sub)
        axB.annotate(f"n={n_val} ({pct_val:.1f}%)", xy=(i + dx, wt), xytext=(0, dy), textcoords="offset points",
                     ha="center", fontsize=6.6, color=cor, fontweight="bold",
                     bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))

# brackets com o p-valor do Mann-Whitney U acima de cada par de caixas
y_topo = 95
y0 = 84
for i, g in enumerate(ordem_grupos):
    p = mw_resultados[g]["p"]
    txt = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    if p < 0.05:
        txt += f" ({estrelas(p)})"
    x1, x2 = i - 0.18, i + 0.18
    axB.plot([x1, x1, x2, x2], [y0, y0 + 2.5, y0 + 2.5, y0], lw=0.9, color="#555555", clip_on=False)
    axB.text((x1 + x2) / 2, y0 + 3.3, txt, ha="center", va="bottom", fontsize=7.6, color="#333333")

axB.set_ylim(0, y_topo)
axB.text(-0.09, 1.05, "B)", transform=axB.transAxes, fontsize=15, fontweight="bold")

# ----------------------------------------------------------------------
for ext in ["png", "pdf", "svg", "tiff"]:
    out_path = os.path.join(OUT_DIR, f"painel_LOS_idade_PT.{ext}")
    fig.savefig(out_path, dpi=300 if ext in ("png", "tiff") else None, bbox_inches="tight")
    print("salvo:", out_path)

plt.show()
plt.close(fig)
