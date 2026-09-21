"""
Probabilidade prevista de óbito por idade (contínua), Vacinado vs.
Não vacinado, restrito aos pacientes com problema respiratório
(Prob_Resp = 1). Mesmo método do script 16 (Óbito ~ Idade * Vacinado,
com IC 95%), aplicado apenas a este subgrupo clínico.
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

GREEN = "#1D9E75"   # Vaccinated
ORANGE = "#E07B39"  # Not vaccinated

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

modelo = smf.glm("Óbito ~ Idade * Vacinado", data=df, family=sm.families.Binomial()).fit()
p_interacao = modelo.pvalues["Idade:Vacinado"]
p_vacinado = modelo.pvalues["Vacinado"]

print(modelo.summary())
print(f"\nSubgrupo: pacientes com problema respiratório (n={n_total})")
print(f"Interação Idade:Vacinado -> p = {p_interacao:.4f}")
print(f"Vacinado (efeito principal) -> p = {p_vacinado:.4f}")

faixas = {
    0: (int(df.loc[df["Vacinado"] == 0, "Idade"].min()), int(df.loc[df["Vacinado"] == 0, "Idade"].max())),
    1: (int(df.loc[df["Vacinado"] == 1, "Idade"].min()), int(df.loc[df["Vacinado"] == 1, "Idade"].max())),
}
curvas = {}
for v in [0, 1]:
    lo, hi = faixas[v]
    grid = pd.DataFrame({"Idade": np.arange(lo, hi + 1), "Vacinado": v})
    pred = modelo.get_prediction(grid).summary_frame(alpha=0.05)
    pred["Idade"] = grid["Idade"].values
    curvas[v] = pred

df["faixa10"] = (df["Idade"] // 10 * 10).astype(int)
emp = df.groupby(["faixa10", "Vacinado"])["Óbito"].agg(["mean", "size"]).reset_index()

# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for v, cor, label in [(0, ORANGE, "Não vacinado"), (1, GREEN, "Vacinado")]:
    pred = curvas[v]
    n = int((df["Vacinado"] == v).sum())
    ax.plot(pred["Idade"], 100 * pred["mean"], color=cor, linewidth=2.2, label=f"{label} (n={n})", zorder=4)
    ax.fill_between(pred["Idade"], 100 * pred["mean_ci_lower"], 100 * pred["mean_ci_upper"],
                     color=cor, alpha=0.18, zorder=2)

    sub_emp = emp[emp["Vacinado"] == v]
    ax.scatter(sub_emp["faixa10"] + 5, 100 * sub_emp["mean"],
               s=sub_emp["size"] * 2.2, color=cor, edgecolor="white", linewidth=0.8,
               alpha=0.85, zorder=5)

ax.set_xlabel("Idade", fontsize=11)
ax.set_ylabel("Probabilidade prevista de óbito (%)", fontsize=11)
ax.set_title("Vacinação protege por idade em quem tem problema respiratório?\n"
             "Probabilidade prevista de óbito por idade, Vacinado vs. Não vacinado (apenas Prob_Resp=1)",
             fontsize=13, fontweight="bold", color="#333333", pad=14)
ax.set_xlim(0, 97)
ax.set_ylim(0, 100)
ax.grid(axis="both", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(title="Status vacinal", loc="upper left", frameon=True, framealpha=0.9, fontsize=10, title_fontsize=10.5)

txt_p = "p < 0.001" if p_interacao < 0.001 else f"p = {p_interacao:.3f}"
sig = " (protege de forma diferente por idade)" if p_interacao < 0.05 else " (sem diferença estatística por idade)"
ax.text(0.98, 0.03, f"Interação Idade × Vacinado: {txt_p}{sig}",
        transform=ax.transAxes, fontsize=8.8, va="bottom", ha="right", color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#cccccc", lw=0.7))

fig.text(0.01, 0.005,
          f"Subgrupo: pacientes com problema respiratório (Prob_Resp=1), n={n_total}. "
          "Linhas = probabilidade prevista (Óbito ~ Idade × Vacinado), sombra = IC 95%. "
          "Pontos = taxa de óbito observada por década de idade (tamanho ~ n); curva do vacinado restrita à faixa etária com dados.",
          ha="left", va="bottom", fontsize=7.6, color="#555555")

fig.subplots_adjust(top=0.87, bottom=0.11, left=0.09, right=0.97)

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(BASE_DIR, f"impacto_vacina_prob_resp_por_idade.{ext}")
    fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
