import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "lineplot_doses_outcome.png")

DOSE_LABELS = {0: "No dose", 1: "1 dose", 2: "2 doses",
               3: "3 doses", 4: "4 doses"}

# ── Palette ───────────────────────────────────────────────────────────────
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_ALTA = "#1D9E75"   # discharge (good outcome)
COR_OBITO = "#E0523F"  # death (critical outcome)

# ════════════════════════════════════════════════════════════
# 1. LOAD AND AGGREGATE
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)
tab = pd.crosstab(dados["Vacinas"], dados["Óbito"]).reindex(
    sorted(DOSE_LABELS), fill_value=0)
tab.columns = ["Alta", "Obito"]

x = list(range(len(tab)))
labels = [DOSE_LABELS[d] for d in tab.index]
n_total = len(dados)
n_deaths = int(dados["Óbito"].sum())

# ════════════════════════════════════════════════════════════
# 2. LINE CHART
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BG)
ax.set_facecolor(PANEL)

ax.yaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.plot(x, tab["Alta"], color=COR_ALTA, linewidth=2.5, marker="o",
        markersize=8, markerfacecolor=COR_ALTA, markeredgecolor=BG,
        markeredgewidth=1.5, zorder=3, label="Discharge (Óbito = 0)")
ax.plot(x, tab["Obito"], color=COR_OBITO, linewidth=2.5, marker="o",
        markersize=8, markerfacecolor=COR_OBITO, markeredgecolor=BG,
        markeredgewidth=1.5, zorder=3, label="Death (Óbito = 1)")

# Direct value labels above/below each point
for xi, (a, o) in zip(x, zip(tab["Alta"], tab["Obito"])):
    ax.annotate(str(a), (xi, a), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=11, color=COR_ALTA, fontweight="bold")
    ax.annotate(str(o), (xi, o), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=11, color=COR_OBITO, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, tab.sum(axis=1))],
                    fontsize=12, color=TEXT)
ax.tick_params(axis="y", colors=SUBTEXT, labelsize=11)
ax.tick_params(axis="x", length=0, pad=12)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylabel("Number of patients", fontsize=13, color=SUBTEXT, labelpad=8)
ax.set_ylim(-tab.values.max() * 0.05, tab.values.max() * 1.18)

ax.legend(fontsize=11, frameon=True, edgecolor=BORDER, facecolor=BG,
          labelcolor=TEXT, loc="upper right", framealpha=0.97)

fig.text(0.5, 0.985, "Discharges and Deaths by Number of Vaccine Doses",
          ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.5, 0.945, f"Overall n = {n_total} | Deaths n = {n_deaths}",
          ha="center", va="top", fontsize=12, color=SUBTEXT)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Chart saved to: {OUTPUT_PNG}")
print(tab)
plt.show()
