"""
Probabilidade prevista de óbito por idade (contínua), comparando
presença vs. ausência de cada condição, em múltiplos painéis:
Vacinação, Problema respiratório, Choque, Câncer.

Cada painel ajusta Óbito ~ Idade * <variável> e plota a curva prevista
(com IC 95%) por idade para cada nível da variável, testando se o
efeito daquela condição sobre o óbito muda conforme a idade
(interação Idade × variável).

Observação: Sepse e Cuidados Paliativos NÃO existem como colunas em
New_pacientes703.xlsx (existiam no dataset antigo 703pacientes.xlsx
usado no script 05) e por isso não entram neste painel.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

GREEN = "#1D9E75"    # lower-risk reference group (not vaccinated... no: absence of condition / vaccinated)
ORANGE = "#E07B39"   # higher-risk / condition-present group

plt.rcParams.update({
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})

df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

# (coluna, título do painel, rótulo grupo 0, rótulo grupo 1,
#  cor grupo 0, cor grupo 1)
PAINEIS = [
    ("Vacinado", "Vacinação", "Não vacinado", "Vacinado", ORANGE, GREEN),
    ("Prob_Resp", "Problema respiratório", "Sem problema resp.", "Com problema resp.", GREEN, ORANGE),
    ("Choques", "Choque", "Sem choque", "Com choque", GREEN, ORANGE),
    ("Cancer", "Condição oncológica", "Sem câncer", "Com câncer", GREEN, ORANGE),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
axes = axes.flatten()

for ax, (col, titulo, lbl0, lbl1, cor0, cor1) in zip(axes, PAINEIS):
    modelo = smf.glm(f"Óbito ~ Idade * {col}", data=df, family=sm.families.Binomial()).fit()
    p_int = modelo.pvalues[f"Idade:{col}"]

    for v, cor, lbl in [(0, cor0, lbl0), (1, cor1, lbl1)]:
        sub_idade = df.loc[df[col] == v, "Idade"]
        lo, hi = int(sub_idade.min()), int(sub_idade.max())
        grid = pd.DataFrame({"Idade": np.arange(lo, hi + 1), col: v})
        pred = modelo.get_prediction(grid).summary_frame(alpha=0.05)
        pred["Idade"] = grid["Idade"].values

        n = len(sub_idade)
        ax.plot(pred["Idade"], 100 * pred["mean"], color=cor, linewidth=2.0,
                label=f"{lbl} (n={n})", zorder=4)
        ax.fill_between(pred["Idade"], 100 * pred["mean_ci_lower"], 100 * pred["mean_ci_upper"],
                         color=cor, alpha=0.16, zorder=2)

        emp = df[df[col] == v].copy()
        emp["faixa10"] = (emp["Idade"] // 10 * 10).astype(int)
        emp_g = emp.groupby("faixa10")["Óbito"].agg(["mean", "size"]).reset_index()
        ax.scatter(emp_g["faixa10"] + 5, 100 * emp_g["mean"], s=emp_g["size"] * 1.8,
                   color=cor, edgecolor="white", linewidth=0.7, alpha=0.8, zorder=5)

    txt_p = "p < 0.001" if p_int < 0.001 else f"p = {p_int:.3f}"
    sig = " *" if p_int < 0.05 else " (ns)"
    ax.text(0.98, 0.03, f"Interação Idade×{col.replace('_', ' ')}: {txt_p}{sig}",
            transform=ax.transAxes, fontsize=7.6, va="bottom", ha="right", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc", lw=0.6))

    ax.set_title(titulo, fontsize=12.5, fontweight="bold", color="#333333", pad=8)
    ax.set_xlabel("Idade", fontsize=10)
    ax.set_ylabel("Probabilidade prevista de óbito (%)", fontsize=10)
    ax.set_xlim(0, 97)
    ax.set_ylim(0, 100)
    ax.grid(axis="both", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)

fig.suptitle("Probabilidade prevista de óbito por idade, por condição clínica presente vs. ausente",
             fontsize=15, fontweight="bold", color="#333333", y=0.995)
fig.text(0.5, 0.965,
          "Sepse e Cuidados Paliativos não existem como colunas nesta base (New_pacientes703.xlsx) e não puderam ser incluídos.",
          ha="center", va="top", fontsize=9, color="#a85320", style="italic")
fig.text(0.01, 0.005,
          "Linhas = probabilidade prevista (Óbito ~ Idade × condição), sombra = IC 95%. "
          "Pontos = taxa de óbito observada por década de idade (tamanho ~ n). * = interação Idade×condição significativa (p<0,05).",
          ha="left", va="bottom", fontsize=7.6, color="#555555")

fig.tight_layout(rect=[0, 0.02, 1, 0.94])

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(BASE_DIR, f"impacto_condicoes_por_idade.{ext}")
    fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
