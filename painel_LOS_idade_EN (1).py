"""
Panel A) Trend in Length of Stay by Age
Panel B) Length of Stay by Age Group
Source: New_pacientes703.xlsx (curated cohort, n=703)

Statistical analysis: Shapiro-Wilk normality test (to justify the
non-parametric approach) + Mann-Whitney U test (Death vs Discharge
comparison). The statistical annotations are NOT drawn as a text box in
the figure footer; the full results are printed to the console for use
in text/notes.
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
# Font (Arial not available in this environment -> Liberation Sans,
# metric-compatible with Arial; portable fallback)
# ----------------------------------------------------------------------
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

GREEN = "#1D9E75"   # Discharge
ORANGE = "#E07B39"  # Death

plt.rcParams.update({
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
df["Obito_lbl"] = df["Óbito"].map({0: "Discharge", 1: "Death"})
LABELS_IDADE_CAT = {0: "0-18", 1: "19-40", 2: "41-60", 3: "+60"}
df["Grupo_etario"] = df["Idade_cat"].map(LABELS_IDADE_CAT)
ordem_grupos = ["0-18", "19-40", "41-60", "+60"]

# ----------------------------------------------------------------------
# Sample sizes (n) by age, age group, and outcome
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
# STATISTICAL ANALYSIS: Shapiro-Wilk (normality) + Mann-Whitney U
# ========================================================================

def estrelas(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

# --- Shapiro-Wilk by outcome (overall) ---
sw_disch = stats.shapiro(df[df["Óbito"] == 0]["Dias_permanência"])
sw_death = stats.shapiro(df[df["Óbito"] == 1]["Dias_permanência"])

# --- Global Mann-Whitney U (Discharge vs Death, corresponds to Panel A) ---
mw_global = stats.mannwhitneyu(
    df[df["Óbito"] == 1]["Dias_permanência"],
    df[df["Óbito"] == 0]["Dias_permanência"],
    alternative="two-sided",
)

# --- Mann-Whitney U by age group (Panel B) + Shapiro-Wilk per cell ---
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
    for lbl, serie in [("Discharge", disch), ("Death", death)]:
        sw_por_grupo[g][lbl] = stats.shapiro(serie) if len(serie) >= 3 else None

# --- console output (for copying as text/note) ---
print("=" * 70)
print("SHAPIRO-WILK (normality of the LOS distribution, by outcome)")
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
print(f"Global (all ages, Panel A): U = {mw_global.statistic:.1f}, "
      f"p {'< 0.001' if p_glob < 0.001 else '= ' + format(p_glob, '.4f')} ({estrelas(p_glob)})")
for g in ordem_grupos:
    r = mw_resultados[g]
    txt_p = "< 0.001" if r["p"] < 0.001 else f"= {r['p']:.4f}"
    print(f"{g}: Death md={r['med_death']} (n={r['n_death']}) vs Discharge md={r['med_disch']} (n={r['n_disch']}) "
          f"-> p {txt_p} ({estrelas(r['p'])})")
print()

# ========================================================================
# FIGURE (no statistical text boxes in the footer)
# ========================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.6))
fig.subplots_adjust(wspace=0.34, top=0.85, bottom=0.14, left=0.055, right=0.87)

# ---------------------------- PANEL A ------------------------------------
media_idade = (
    df.groupby(["Idade", "Obito_lbl"])["Dias_permanência"]
    .mean()
    .unstack()
    .reindex(range(0, df["Idade"].max() + 1))
)

for col, cor, z in [("Discharge", GREEN, 3), ("Death", ORANGE, 2)]:
    y = media_idade[col].values
    x = media_idade.index.values
    n_col = n_disch_total if col == "Discharge" else n_death_total
    pct_col = 100 * n_col / n_total
    axA.plot(x, y, color=cor, linewidth=1.4, label=f"{col} (n = {n_col}, {pct_col:.1f}%)", zorder=z)
    axA.fill_between(x, y, 0, color=cor, alpha=0.18, zorder=1,
                      where=~np.isnan(y.astype(float)), interpolate=False)

axA.set_title("Trend in Length of Stay by Age", fontsize=14, fontweight="bold", color="#333333", pad=10)
axA.set_xlabel("Age", fontsize=11)
axA.set_ylabel("Length of stay (days)", fontsize=11)
axA.set_xlim(0, df["Idade"].max())
axA.set_ylim(0, None)
axA.grid(axis="both", alpha=0.5)
axA.set_axisbelow(True)

# n markers at every 5-year mark (to avoid cluttering the plot)
for idade_marca in range(0, df["Idade"].max() + 1, 5):
    if idade_marca not in n_por_idade.index:
        continue
    for col, cor, dy in [("Discharge", GREEN, 6), ("Death", ORANGE, -6)]:
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

axA.legend(title="Outcome", loc="upper right", frameon=True, framealpha=0.9, fontsize=9, title_fontsize=9.5)
p_glob_txt = "p < 0.001" if p_glob < 0.001 else f"p = {p_glob:.3f}"
sig_glob = f" ({estrelas(p_glob)})" if p_glob < 0.05 else ""
axA.text(0.012, 0.98, f"Mann-Whitney U (global, all ages): {p_glob_txt}{sig_glob}",
         transform=axA.transAxes, fontsize=8.5, va="top", ha="left", color="#333333",
         bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#cccccc", lw=0.7))
axA.text(-0.09, 1.05, "A)", transform=axA.transAxes, fontsize=15, fontweight="bold")

# ---------------------------- PANEL B -------------------------------------
sns.boxplot(
    data=df, x="Grupo_etario", y="Dias_permanência", hue="Obito_lbl",
    order=ordem_grupos, hue_order=["Discharge", "Death"],
    palette={"Discharge": GREEN, "Death": ORANGE},
    ax=axB, showfliers=False, width=0.6, linewidth=1.0,
    boxprops=dict(alpha=0.9), medianprops=dict(color="#222222", linewidth=1.3)
)
sns.stripplot(
    data=df, x="Grupo_etario", y="Dias_permanência", hue="Obito_lbl",
    order=ordem_grupos, hue_order=["Discharge", "Death"],
    dodge=True, ax=axB, palette={"Discharge": "#4d4d4d", "Death": "#4d4d4d"},
    alpha=0.35, size=3, jitter=0.18, legend=False
)

axB.set_title("Length of Stay by Age Group", fontsize=14, fontweight="bold", color="#333333", pad=10)
axB.set_xlabel("Age Group", fontsize=11)
axB.set_ylabel("Length of stay (days)", fontsize=11)
axB.grid(axis="y", alpha=0.5)
axB.set_axisbelow(True)

# x-axis labels with the total N of each age group (absolute and %)
novos_ticks = [f"{g}\n(N = {int(n_por_grupo[g])}, {100*n_por_grupo[g]/n_total:.1f}%)" for g in ordem_grupos]
axB.set_xticks(range(len(ordem_grupos)))
axB.set_xticklabels(novos_ticks, fontsize=9.5)

handles, labels_ = axB.get_legend_handles_labels()
labels_n = [f"Discharge (n = {n_disch_total}, {100*n_disch_total/n_total:.1f}%)",
            f"Death (n = {n_death_total}, {100*n_death_total/n_total:.1f}%)"]
axB.legend(handles[:2], labels_n, title="Outcome", loc="upper left", bbox_to_anchor=(1.01, 1.02),
           frameon=True, framealpha=0.9, fontsize=9, title_fontsize=9.5, borderaxespad=0)

# n of each box (age group x outcome) - absolute and % within the age group,
# positioned at the top of each box's whisker
def _whisker_top(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    lim = q3 + 1.5 * iqr
    dentro = serie[serie <= lim]
    return dentro.max() if len(dentro) else serie.max()

for i, g in enumerate(ordem_grupos):
    n_grupo = int(n_por_grupo[g])
    for dx, dy, col, cor in [(-0.20, 4, "Discharge", "#0f6b4c"), (0.20, 12, "Death", "#a85320")]:
        sub = df[(df["Grupo_etario"] == g) & (df["Obito_lbl"] == col)]["Dias_permanência"]
        n_val = int(n_por_grupo_desfecho.loc[g, col])
        pct_val = 100 * n_val / n_grupo
        wt = _whisker_top(sub)
        axB.annotate(f"n={n_val} ({pct_val:.1f}%)", xy=(i + dx, wt), xytext=(0, dy), textcoords="offset points",
                     ha="center", fontsize=6.6, color=cor, fontweight="bold",
                     bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))

# brackets with the Mann-Whitney U p-value above each pair of boxes
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
    out_path = os.path.join(OUT_DIR, f"painel_LOS_idade_EN.{ext}")
    fig.savefig(out_path, dpi=300 if ext in ("png", "tiff") else None, bbox_inches="tight")
    print("saved:", out_path)

plt.show()
plt.close(fig)
