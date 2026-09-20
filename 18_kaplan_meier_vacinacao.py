import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────
# Curva de sobrevida (Kaplan-Meier) durante a internação, comparando
# pacientes não vacinados, com vacinação incompleta (1 dose) e com
# vacinação completa (2+ doses). Tempo = dias de permanência hospitalar;
# evento = óbito (alta é tratada como censura).
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "kaplan_meier_vacinacao.png")

# ── Paleta ────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_NAO_VAC = "#E0523F"     # não vacinados (crítico)
COR_COMPLETA = "#1D9E75"    # vacinação completa (protetor, em destaque)
COR_INCOMPLETA = "#9AA3AC"  # 1 dose (referência secundária, discreta)


# ════════════════════════════════════════════════════════════
# 1. KAPLAN-MEIER (implementação manual — produto-limite + Greenwood)
# ════════════════════════════════════════════════════════════
def kaplan_meier(duracoes, eventos):
    """Estimador de Kaplan-Meier com erro-padrão de Greenwood.

    Retorna um DataFrame com o tempo, nº em risco, nº de eventos,
    S(t) e o erro-padrão de S(t) em cada tempo de evento observado.
    """
    dados = pd.DataFrame({"t": duracoes, "e": eventos})
    tempos = np.sort(dados["t"].unique())
    n = len(dados)

    linhas = [(0.0, n, 0, 1.0, 0.0)]
    surv = 1.0
    var_soma = 0.0
    for t in tempos:
        n_t = (dados["t"] >= t).sum()
        d_t = ((dados["t"] == t) & (dados["e"] == 1)).sum()
        if n_t == 0:
            continue
        if d_t > 0:
            surv *= (1 - d_t / n_t)
            if n_t > d_t:
                var_soma += d_t / (n_t * (n_t - d_t))
        se = surv * np.sqrt(var_soma)
        linhas.append((t, n_t, d_t, surv, se))

    return pd.DataFrame(linhas, columns=["t", "n_risco", "d_eventos", "S", "se"])


def logrank_test(duracoes_a, eventos_a, duracoes_b, eventos_b):
    """Teste de log-rank (2 grupos) — estatística qui-quadrado, 1 gl."""
    df_a = pd.DataFrame({"t": duracoes_a, "e": eventos_a, "g": "a"})
    df_b = pd.DataFrame({"t": duracoes_b, "e": eventos_b, "g": "b"})
    dados = pd.concat([df_a, df_b], ignore_index=True)

    tempos_evento = np.sort(dados.loc[dados["e"] == 1, "t"].unique())
    obs_a = esp_a = var_soma = 0.0
    for t in tempos_evento:
        em_risco = dados[dados["t"] >= t]
        n = len(em_risco)
        n_a = (em_risco["g"] == "a").sum()
        d = (em_risco.loc[em_risco["t"] == t, "e"] == 1).sum()
        d_a = ((em_risco["g"] == "a") & (em_risco["t"] == t) &
               (em_risco["e"] == 1)).sum()
        if n <= 1 or n == 0:
            continue
        e_a = n_a * d / n
        v = (n_a * (n - n_a) * d * (n - d)) / (n ** 2 * (n - 1)) if n > 1 else 0
        obs_a += d_a
        esp_a += e_a
        var_soma += v

    chi2 = (obs_a - esp_a) ** 2 / var_soma if var_soma > 0 else 0.0
    return chi2


from scipy.stats import chi2 as chi2_dist  # noqa: E402


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


# ════════════════════════════════════════════════════════════
# 2. CARREGAR DADOS E DEFINIR GRUPOS DE VACINAÇÃO
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)


def classifica(vacinas):
    if vacinas == 0:
        return "Não vacinados"
    if vacinas == 1:
        return "1 dose (incompleta)"
    return "Vacinação completa (2+ doses)"


dados["Grupo"] = dados["Vacinas"].apply(classifica)

n_total = len(dados)
n_obitos = int(dados["Óbito"].sum())

grupos_info = {}
for grupo in ["Não vacinados", "1 dose (incompleta)", "Vacinação completa (2+ doses)"]:
    sub = dados[dados["Grupo"] == grupo]
    grupos_info[grupo] = {
        "km": kaplan_meier(sub["Dias_permanência"], sub["Óbito"]),
        "n": len(sub),
        "obitos": int(sub["Óbito"].sum()),
    }

# Teste de log-rank: vacinação completa vs. não vacinados (comparação principal)
sub_nao_vac = dados[dados["Grupo"] == "Não vacinados"]
sub_completa = dados[dados["Grupo"] == "Vacinação completa (2+ doses)"]
chi2_lr = logrank_test(sub_nao_vac["Dias_permanência"], sub_nao_vac["Óbito"],
                        sub_completa["Dias_permanência"], sub_completa["Óbito"])
p_lr = chi2_dist.sf(chi2_lr, df=1)

# ════════════════════════════════════════════════════════════
# 3. GRÁFICO — curvas de sobrevida em degraus, com IC de Greenwood
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG)
ax.set_facecolor(PANEL)
ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

estilos = {
    "Não vacinados": dict(color=COR_NAO_VAC, lw=2.6, ls="-", zorder=4, alpha=0.95),
    "1 dose (incompleta)": dict(color=COR_INCOMPLETA, lw=1.8, ls=":", zorder=2, alpha=0.9),
    "Vacinação completa (2+ doses)": dict(color=COR_COMPLETA, lw=2.8, ls="-", zorder=5, alpha=0.95),
}

for grupo, estilo in estilos.items():
    km = grupos_info[grupo]["km"]
    ax.step(km["t"], km["S"], where="post", label=None, **estilo)
    # Banda de confiança de Greenwood apenas para os 2 grupos principais
    if grupo != "1 dose (incompleta)":
        lo = np.clip(km["S"] - 1.96 * km["se"], 0, 1)
        hi = np.clip(km["S"] + 1.96 * km["se"], 0, 1)
        ax.fill_between(km["t"], lo, hi, step="post",
                         color=estilo["color"], alpha=0.15, zorder=1)

ax.set_xlabel("Dias de internação", fontsize=13, color=SUBTEXT, labelpad=8)
ax.set_ylabel("Probabilidade de sobrevida (S(t))", fontsize=13, color=SUBTEXT,
              labelpad=8)
ax.set_ylim(0, 1.03)
xmax = dados["Dias_permanência"].quantile(0.98)
ax.set_xlim(0, xmax)
ax.tick_params(axis="both", colors=SUBTEXT, labelsize=11)

legend_labels = [
    f"Vacinação completa, 2+ doses (n={grupos_info['Vacinação completa (2+ doses)']['n']}, "
    f"óbitos={grupos_info['Vacinação completa (2+ doses)']['obitos']})",
    f"Não vacinados (n={grupos_info['Não vacinados']['n']}, "
    f"óbitos={grupos_info['Não vacinados']['obitos']})",
    f"1 dose / incompleta (n={grupos_info['1 dose (incompleta)']['n']}, "
    f"óbitos={grupos_info['1 dose (incompleta)']['obitos']}) — referência, "
    "não comparada",
]
from matplotlib.lines import Line2D  # noqa: E402
legend_elements = [
    Line2D([0], [0], color=COR_COMPLETA, lw=2.8, label=legend_labels[0]),
    Line2D([0], [0], color=COR_NAO_VAC, lw=2.6, label=legend_labels[1]),
    Line2D([0], [0], color=COR_INCOMPLETA, lw=1.8, linestyle=":",
           label=legend_labels[2]),
]
ax.legend(handles=legend_elements, fontsize=10, frameon=True, edgecolor=BORDER,
          facecolor=BG, labelcolor=TEXT, loc="lower left", framealpha=0.97)

# Anotação do teste de log-rank (comparação principal)
ax.text(0.99, 0.98, f"Log-rank (completa vs. não vacinados)\n"
                     f"χ²={chi2_lr:.2f}, gl=1, {formata_p(p_lr)}",
        transform=ax.transAxes, fontsize=10, color="#333333",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray",
                  alpha=0.95, linewidth=0.8))

fig.text(0.5, 0.985,
          "Sobrevida Durante a Internação por Status Vacinal",
          ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.5, 0.945,
          f"n total = {n_total} | Óbitos = {n_obitos} | "
          "Tempo = dias de permanência | evento = óbito (alta = censura)",
          ha="center", va="top", fontsize=11.5, color=SUBTEXT)
fig.text(0.13, 0.02,
          "Pacientes com 1 dose (vacinação incompleta) são mostrados apenas "
          "como referência — sujeitos a viés de indicação (vacinação "
          "tardia/emergencial durante a internação) e excluídos da "
          "comparação estatística principal.",
          ha="left", va="bottom", fontsize=9, style="italic", color=SUBTEXT)

plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
print(f"Log-rank completa vs. não vacinados: chi2={chi2_lr:.3f}, p={p_lr:.4f}")
plt.show()
