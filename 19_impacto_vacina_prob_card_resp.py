"""
Impacto da vacinação (Vacinado 0 vs 1) no desfecho (Óbito vs Alta),
por problema cardiovascular (Prob_Card), restrito aos pacientes com
problema respiratório (Prob_Resp = 1).

Substitui a idade (usada em 18_impacto_vacina_prob_resp_por_idade.py)
pela variável de problema cardiovascular como eixo de estratificação,
já que Prob_Card é binária (Sem/Com), no mesmo formato de barras 100%
empilhadas + Fisher exato usado em 17_impacto_vacina_por_condicao.py.
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
df_all = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
df = df_all[df_all["Prob_Resp"] == 1].copy()
n_total = len(df)

linhas = []
for cat, lbl in [(0, "Sem"), (1, "Com")]:
    sub = df[df["Prob_Card"] == cat]
    linha = {"cat": cat, "cat_lbl": lbl}
    for v in [0, 1]:
        s = sub[sub["Vacinado"] == v]
        n = len(s)
        obitos = int(s["Óbito"].sum())
        linha[f"n_{v}"] = n
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
    linhas.append(linha)
res = pd.DataFrame(linhas)

# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 6.5))

n_cats = len(res)
x = np.arange(n_cats)
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
        po, pa = pct_obito[i], pct_alta[i]
        if pa >= 3:
            ax.text(xi, pa / 2, f"{pa:.1f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        if po >= 3:
            ax.text(xi, pa + po / 2, f"{po:.1f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax.text(xi, -3, f"{vlbl}\n(n={n})", ha="center", va="top", fontsize=8, color="#333333")

for i, xi in enumerate(x):
    p = res.loc[i, "p"]
    if np.isnan(p):
        continue
    txt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    sig = " *" if p < 0.05 else ""
    x1, x2 = xi + offsets[0], xi + offsets[1]
    y0 = 103
    ax.plot([x1, x1, x2, x2], [y0, y0 + 1.5, y0 + 1.5, y0], lw=0.9, color="#555555", clip_on=False)
    ax.text((x1 + x2) / 2, y0 + 2.3, f"{txt}{sig}", ha="center", va="bottom", fontsize=8.5, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels([f"\n\n{res.loc[i, 'cat_lbl']} problema cardiovascular" for i in range(n_cats)],
                    fontsize=10.5, fontweight="bold")
ax.set_ylabel("Percentual de pacientes (%)", fontsize=11)
ax.set_ylim(0, 112)
ax.set_title("Impacto da vacinação no desfecho, por problema cardiovascular\n"
             "(restrito a pacientes com problema respiratório)",
             fontsize=13, fontweight="bold", color="#333333", pad=14)
ax.grid(axis="y", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, framealpha=0.9,
          fontsize=9.5, title="Desfecho", title_fontsize=10)

fig.text(0.01, 0.005,
          f"Subgrupo: pacientes com problema respiratório (Prob_Resp=1), n={n_total}. "
          "* = diferença estatisticamente significativa (Fisher exato, p<0,05) entre vacinados e não vacinados dentro daquele estrato.",
          ha="left", va="bottom", fontsize=7.8, color="#555555")

fig.subplots_adjust(top=0.85, bottom=0.13, left=0.12, right=0.78)

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(BASE_DIR, f"impacto_vacina_prob_card_resp.{ext}")
    fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
