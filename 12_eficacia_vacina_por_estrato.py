"""
Eficácia da vacinação (Vacinado 1 vs 0) na redução do óbito,
estratificada por cada categoria das demais colunas clínicas/demográficas.

Redução (p.p.) = taxa de óbito (não vacinados) - taxa de óbito (vacinados)
dentro de cada estrato. Valor positivo = vacinados com menor mortalidade.
Significância: teste exato de Fisher (2x2) por estrato.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "New_pacientes703.xlsx")

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

GREEN = "#1D9E75"   # reduction in death among vaccinated
ORANGE = "#E07B39"  # increase in death among vaccinated
N_MIN = 10          # minimum n per arm (vac / not vac) within a stratum

plt.rcParams.update({
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})

# ----------------------------------------------------------------------
# Data & stratified analysis
# ----------------------------------------------------------------------
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

exclude = {"Registro", "Atend_Profissão", "Data de Entrada", "Vacinas", "Vacinado",
           "Óbito", "Dias_permanência", "Contagem_profissão", "Idade"}
cols = [c for c in df.columns if c not in exclude]

rows = []
for col in cols:
    cats = sorted(df[col].dropna().unique())
    if len(cats) > 6:
        continue
    for cat in cats:
        sub = df[df[col] == cat]
        vac1 = sub[sub["Vacinado"] == 1]
        vac0 = sub[sub["Vacinado"] == 0]
        n1, n0 = len(vac1), len(vac0)
        if n1 < N_MIN or n0 < N_MIN:
            continue
        d1, d0 = vac1["Óbito"].sum(), vac0["Óbito"].sum()
        rate1, rate0 = d1 / n1, d0 / n0
        arr = rate0 - rate1
        _, p = stats.fisher_exact([[d1, n1 - d1], [d0, n0 - d0]])
        rows.append(dict(estrato=f"{col} = {int(cat) if float(cat).is_integer() else cat}",
                          n_vac=n1, n_nvac=n0, taxa_vac=rate1, taxa_nvac=rate0,
                          arr_pp=100 * arr, p=p))

res = pd.DataFrame(rows).sort_values("arr_pp", ascending=True).reset_index(drop=True)

# ----------------------------------------------------------------------
# Figure: diverging bar chart
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 0.42 * len(res) + 2.4))

y = np.arange(len(res))
cores = [GREEN if v >= 0 else ORANGE for v in res["arr_pp"]]
alphas = [0.95 if p < 0.05 else 0.45 for p in res["p"]]

bars = ax.barh(y, res["arr_pp"], color=cores, height=0.62, zorder=3)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)

ax.axvline(0, color="#444444", linewidth=0.9, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(res["estrato"], fontsize=8.8)
ax.set_xlabel("Redução do óbito entre vacinados vs. não vacinados (pontos percentuais)", fontsize=10)
ax.set_title("Onde a vacinação foi associada a menor mortalidade\n(estratificado por variável clínica/demográfica; Fisher exato por estrato)",
             fontsize=12.5, fontweight="bold", color="#333333", pad=12)
ax.grid(axis="x", alpha=0.5)
ax.set_axisbelow(True)

# direct labels: value + significance + n, aligned in a fixed column to the
# right of every bar (regardless of sign) so long negative-bar labels never
# creep back toward the y-axis tick labels.
xmax = res["arr_pp"].abs().max()
x_label_col = xmax + 4
for yi, (_, r) in zip(y, res.iterrows()):
    txt_p = "p<0.001" if r["p"] < 0.001 else f"p={r['p']:.3f}"
    sig = " *" if r["p"] < 0.05 else ""
    label = f"{r['arr_pp']:+.1f} p.p. ({txt_p}{sig})  [vac n={r['n_vac']}, não-vac n={r['n_nvac']}]"
    ax.text(x_label_col, yi, label, va="center", ha="left", fontsize=7.3, color="#333333")

ax.set_xlim(-xmax - 4, xmax + 42)

legend_txt = ("Verde = vacinados com menor mortalidade no estrato   |   "
              "Laranja = vacinados com maior mortalidade no estrato   |   "
              "* = estatisticamente significativo (Fisher, p<0.05); opacidade reduzida = não significativo")
fig.text(0.5, 0.005, legend_txt, ha="center", va="bottom", fontsize=8, color="#555555", wrap=True)

fig.subplots_adjust(top=0.94, bottom=0.08, left=0.22, right=0.97)
for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(BASE_DIR, f"eficacia_vacina_por_estrato.{ext}")
    fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
