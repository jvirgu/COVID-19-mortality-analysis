"""
Impacto da vacinação (Vacinado 0 vs 1) no desfecho (Óbito vs Alta),
dentro de cada faixa etária.

Barras 100% empilhadas: Alta (verde) + Óbito (laranja), uma barra para
não vacinados e uma para vacinados, por faixa etária. Comparação
estatística (Óbito vacinado vs não vacinado) por teste exato de Fisher,
dentro de cada faixa etária.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

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

GREEN = "#1D9E75"   # Discharge
ORANGE = "#E07B39"  # Death

plt.rcParams.update({
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})

# ----------------------------------------------------------------------
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
LABELS_IDADE_CAT = {0: "0-18", 1: "19-40", 2: "41-60", 3: "+60"}
df["Grupo_etario"] = df["Idade_cat"].map(LABELS_IDADE_CAT)
df["Vac_lbl"] = df["Vacinado"].map({0: "Não vacinado", 1: "Vacinado"})
ordem_grupos = ["0-18", "19-40", "41-60", "+60"]

registros = []
for g in ordem_grupos:
    sub = df[df["Grupo_etario"] == g]
    linha = {"grupo": g}
    for v, vlbl in [(0, "Não vacinado"), (1, "Vacinado")]:
        s = sub[sub["Vacinado"] == v]
        n = len(s)
        obitos = int(s["Óbito"].sum())
        altas = n - obitos
        linha[f"n_{v}"] = n
        linha[f"obitos_{v}"] = obitos
        linha[f"altas_{v}"] = altas
        linha[f"pct_obito_{v}"] = 100 * obitos / n if n else np.nan
    v1 = sub[sub["Vacinado"] == 1]
    v0 = sub[sub["Vacinado"] == 0]
    if len(v1) >= 5 and len(v0) >= 5:
        _, p = stats.fisher_exact([
            [v1["Óbito"].sum(), len(v1) - v1["Óbito"].sum()],
            [v0["Óbito"].sum(), len(v0) - v0["Óbito"].sum()],
        ])
    else:
        p = np.nan
    linha["p"] = p
    registros.append(linha)

res = pd.DataFrame(registros)

# ----------------------------------------------------------------------
# Figure: 100% stacked bars, grouped by age group, split by vaccination
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

n_grupos = len(ordem_grupos)
x = np.arange(n_grupos)
width = 0.32
offsets = {0: -width / 2 - 0.02, 1: width / 2 + 0.02}

for v, vlbl in [(0, "Não vacinado"), (1, "Vacinado")]:
    pct_obito = res[f"pct_obito_{v}"].values
    pct_alta = 100 - pct_obito
    xs = x + offsets[v]
    ax.bar(xs, pct_alta, width=width, color=GREEN, alpha=0.9, zorder=3,
           label="Alta" if v == 0 else None, edgecolor="white", linewidth=0.6)
    ax.bar(xs, pct_obito, width=width, bottom=pct_alta, color=ORANGE, alpha=0.9, zorder=3,
           label="Óbito" if v == 0 else None, edgecolor="white", linewidth=0.6)

    for i, xi in enumerate(xs):
        n = res.loc[i, f"n_{v}"]
        po = pct_obito[i]
        pa = pct_alta[i]
        if pa >= 3:
            ax.text(xi, pa / 2, f"{pa:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if po >= 3:
            ax.text(xi, pa + po / 2, f"{po:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(xi, -3, f"{vlbl}\n(n={n})", ha="center", va="top", fontsize=7.8, color="#333333")

# p-value bracket between the two bars of each age group
for i, xi in enumerate(x):
    p = res.loc[i, "p"]
    if np.isnan(p):
        continue
    txt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    sig = " *" if p < 0.05 else ""
    x1, x2 = xi + offsets[0], xi + offsets[1]
    y0 = 103
    ax.plot([x1, x1, x2, x2], [y0, y0 + 1.5, y0 + 1.5, y0], lw=0.9, color="#555555", clip_on=False)
    ax.text((x1 + x2) / 2, y0 + 2.3, f"{txt}{sig}", ha="center", va="bottom", fontsize=7.8, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels([f"\n\n{g}" for g in ordem_grupos], fontsize=11, fontweight="bold")
ax.set_ylabel("Percentual de pacientes (%)", fontsize=11)
ax.set_ylim(0, 112)
ax.set_title("Impacto da vacinação no desfecho (Óbito x Alta) por faixa etária",
             fontsize=13.5, fontweight="bold", color="#333333", pad=14)
ax.grid(axis="y", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, framealpha=0.9,
          fontsize=9.5, title="Desfecho", title_fontsize=10)

fig.text(0.01, 0.005,
          "* = diferença estatisticamente significativa (Fisher exato, p<0,05) entre vacinados e não vacinados na faixa etária. "
          "Atenção: n de vacinados é baixo em 0-18 (n=7) e 19-40/41-60 (n=22-25) — interpretar com cautela.",
          ha="left", va="bottom", fontsize=7.6, color="#555555")

fig.subplots_adjust(top=0.90, bottom=0.16, left=0.08, right=0.82)

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(BASE_DIR, f"impacto_vacina_faixa_etaria.{ext}")
    fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
