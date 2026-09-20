import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "lineplot_ano_doses.png")

DOSE_LABELS = {0: "Nenhuma dose", 1: "1 dose", 2: "2 doses",
               3: "3 doses", 4: "4 doses"}

# ── Paleta ────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_ALTA = "#1D9E75"   # alta (desfecho favorável)
COR_OBITO = "#E0523F"  # óbito (desfecho crítico)

# ════════════════════════════════════════════════════════════
# 1. CARREGAR E AGREGAR OS DADOS
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)
dados["Ano"] = pd.to_datetime(dados["Data de Entrada"]).dt.year

n_total = len(dados)
n_obitos = int(dados["Óbito"].sum())
anos_todos = sorted(dados["Ano"].unique())

# ════════════════════════════════════════════════════════════
# 2. GRÁFICO EM PAINÉIS (pequenos múltiplos) — um por número de doses
# ════════════════════════════════════════════════════════════
doses = sorted(DOSE_LABELS)
fig, axes = plt.subplots(1, len(doses), figsize=(19, 6), facecolor=BG,
                          sharey=True)

for ax, dose in zip(axes, doses):
    sub = dados[dados["Vacinas"] == dose]
    tab = sub.groupby(["Ano", "Óbito"]).size().unstack(fill_value=0)
    anos_com_dados = sorted(sub["Ano"].unique())
    tab = tab.reindex(anos_com_dados, fill_value=0)

    x = tab.index.tolist()
    alta = tab.get(0, pd.Series(0, index=tab.index))
    obito = tab.get(1, pd.Series(0, index=tab.index))

    ax.set_facecolor(PANEL)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.plot(x, alta, color=COR_ALTA, linewidth=2.5, marker="o", markersize=8,
            markerfacecolor=COR_ALTA, markeredgecolor=BG, markeredgewidth=1.5,
            zorder=3, label="Alta (Óbito = 0)")
    ax.plot(x, obito, color=COR_OBITO, linewidth=2.5, marker="o", markersize=8,
            markerfacecolor=COR_OBITO, markeredgecolor=BG, markeredgewidth=1.5,
            zorder=3, label="Óbito (Óbito = 1)")

    for xi, a, o in zip(x, alta, obito):
        ax.annotate(str(a), (xi, a), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, color=COR_ALTA, fontweight="bold")
        ax.annotate(str(o), (xi, o), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=10, color=COR_OBITO, fontweight="bold")

    ax.set_xticks(anos_todos)
    ax.set_xticklabels([str(a) for a in anos_todos], fontsize=11, color=TEXT)
    ax.set_xlim(min(anos_todos) - 0.3, max(anos_todos) + 0.3)
    ax.tick_params(axis="y", colors=SUBTEXT, labelsize=11)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title(f"{DOSE_LABELS[dose]}\n(n={len(sub)})", fontsize=13,
                 fontweight="bold", color=TEXT, pad=10)

axes[0].set_ylabel("Número de pacientes", fontsize=13, color=SUBTEXT, labelpad=8)
ymax = axes[0].get_ylim()[1]
for ax in axes:
    ax.set_ylim(-ymax * 0.05, ymax * 1.1)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, fontsize=11, frameon=True, edgecolor=BORDER,
           facecolor=BG, labelcolor=TEXT, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 0.88), framealpha=0.97)

fig.text(0.5, 1.0, "Altas e Óbitos por Ano — Número de Doses de Vacina",
          ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.5, 0.955, f"n total = {n_total} | Óbitos = {n_obitos}",
          ha="center", va="top", fontsize=12, color=SUBTEXT)

plt.tight_layout(rect=[0, 0, 1, 0.80])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Gráfico salvo em: {OUTPUT_PNG}")
plt.show()
