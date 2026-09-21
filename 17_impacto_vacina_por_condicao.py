"""
Impacto da vacinação (Vacinado 0 vs 1) no desfecho (Óbito vs Alta),
dentro de cada condição clínica: Problema respiratório, Choque e
Condição oncológica (Câncer).

Mesma lógica do script 13 (barras 100% empilhadas por faixa etária),
mas estratificando por condição clínica em vez de idade: para cada
condição, um painel com 4 barras (Sem/Com condição x Não vacinado/
Vacinado), teste exato de Fisher (Vacinado vs Não vacinado) dentro de
cada nível da condição.

Observação: Sepse e Cuidados Paliativos foram pedidos, mas NÃO existem
como colunas nem em New_pacientes703.xlsx nem no dataset "antigo"
703pacientes.xlsx (mesmos 703 pacientes, mesma taxa de óbito global,
sem essas duas variáveis) — por isso não entram neste painel.
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

df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

CONDICOES = [
    ("Prob_Resp", "Problema respiratório", "Sem", "Com"),
    ("Choques", "Choque", "Sem", "Com"),
    ("Cancer", "Condição oncológica", "Sem", "Com"),
]


def calcula_estrato(col, lbl_sem, lbl_com):
    linhas = []
    for cat, lbl in [(0, lbl_sem), (1, lbl_com)]:
        sub = df[df[col] == cat]
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
    return pd.DataFrame(linhas)


fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))

for ax, (col, titulo, lbl_sem, lbl_com) in zip(axes, CONDICOES):
    res = calcula_estrato(col, lbl_sem, lbl_com)
    n_cats = len(res)
    x = np.arange(n_cats)
    width = 0.32
    offsets = {0: -width / 2 - 0.02, 1: width / 2 + 0.02}

    for v, vlbl in [(0, "Não vacinado"), (1, "Vacinado")]:
        pct_obito = res[f"pct_obito_{v}"].values
        pct_alta = 100 - pct_obito
        xs = x + offsets[v]
        ax.bar(xs, pct_alta, width=width, color=GREEN, alpha=0.9, zorder=3,
               label="Alta" if (v == 0 and col == "Prob_Resp") else None,
               edgecolor="white", linewidth=0.6)
        ax.bar(xs, pct_obito, width=width, bottom=pct_alta, color=ORANGE, alpha=0.9, zorder=3,
               label="Óbito" if (v == 0 and col == "Prob_Resp") else None,
               edgecolor="white", linewidth=0.6)

        for i, xi in enumerate(xs):
            n = res.loc[i, f"n_{v}"]
            po, pa = pct_obito[i], pct_alta[i]
            if pa >= 3:
                ax.text(xi, pa / 2, f"{pa:.1f}%", ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
            if po >= 3:
                ax.text(xi, pa + po / 2, f"{po:.1f}%", ha="center", va="center", fontsize=7.6, color="white", fontweight="bold")
            ax.text(xi, -3, f"{vlbl}\n(n={n})", ha="center", va="top", fontsize=7.2, color="#333333")

    for i, xi in enumerate(x):
        p = res.loc[i, "p"]
        if np.isnan(p):
            continue
        txt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        sig = " *" if p < 0.05 else ""
        x1, x2 = xi + offsets[0], xi + offsets[1]
        y0 = 103
        ax.plot([x1, x1, x2, x2], [y0, y0 + 1.5, y0 + 1.5, y0], lw=0.9, color="#555555", clip_on=False)
        ax.text((x1 + x2) / 2, y0 + 2.3, f"{txt}{sig}", ha="center", va="bottom", fontsize=7.2, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"\n\n{res.loc[i, 'cat_lbl']} {titulo.lower()}" for i in range(n_cats)],
                        fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Percentual de pacientes (%)", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_title(titulo, fontsize=12.5, fontweight="bold", color="#333333", pad=10)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, title="Desfecho", loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 0.965), frameon=True, framealpha=0.9,
           fontsize=9.5, title_fontsize=10)

fig.suptitle("Impacto da vacinação no desfecho (Óbito x Alta), dentro de cada condição clínica",
             fontsize=15, fontweight="bold", color="#333333", y=1.10)
fig.text(0.5, 1.055,
          "Sepse e Cuidados Paliativos não existem como colunas em nenhum dos datasets fornecidos (novo ou antigo) e não puderam ser incluídos.",
          ha="center", va="top", fontsize=9.5, color="#a85320", style="italic")
fig.text(0.01, 0.005,
          "* = diferença estatisticamente significativa (Fisher exato, p<0,05) entre vacinados e não vacinados dentro daquele estrato.",
          ha="left", va="bottom", fontsize=7.8, color="#555555")

fig.subplots_adjust(top=0.80, bottom=0.14, left=0.055, right=0.99, wspace=0.28)

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(BASE_DIR, f"impacto_vacina_por_condicao.{ext}")
    fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
