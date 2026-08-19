import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── 1. CARREGAR DADOS ──────────────────────────────────────────────
dfnew = pd.read_excel("New_pacientes703.xlsx")

# ── 2. COLUNAS ──────────────────────────────────────────────
col_vacinas = "Vacinas"
col_desfecho = "Óbito"

# ── 3. PREPARAÇÃO ──────────────────────────────────────────────
dfnew["Data de Entrada"] = pd.to_datetime(dfnew["Data de Entrada"], errors="coerce")
dfnew = dfnew[dfnew["Data de Entrada"].dt.year != 2020]

df = dfnew[[col_vacinas, col_desfecho]].dropna()
df[col_vacinas] = df[col_vacinas].astype(int)

ct = pd.crosstab(df[col_vacinas], df[col_desfecho])
ct.columns = ["Alta (0)", "Óbito (1)"]
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
doses = ct_pct.index.tolist()
x = np.arange(len(doses))

# ── 4. CORES ──────────────────────────────────────────────────
COR_ALTA = "#1D9E75"
COR_OBITO = "#D85A30"
BG = "#ffffff"
PANEL = "#f8f9fa"
BORDER = "#dee2e6"
TEXT = "#212529"
SUBTEXT = "#6c757d"

# ── 5. FIGURA ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax.set_facecolor(PANEL)

# Área — Alta
ax.fill_between(x, ct_pct["Alta (0)"],
                 alpha=0.18, color=COR_ALTA, zorder=2)
ax.plot(x, ct_pct["Alta (0)"],
        color=COR_ALTA, linewidth=2.5, marker="o", markersize=6,
        label="Alta (0)", zorder=3, linestyle="-")

# Área — Óbito
ax.fill_between(x, ct_pct["Óbito (1)"],
                 alpha=0.14, color=COR_OBITO, zorder=2)
ax.plot(x, ct_pct["Óbito (1)"],
        color=COR_OBITO, linewidth=2.5, marker="o", markersize=6,
        label="Óbito (1)", zorder=3, linestyle=(0, (6, 3)))

# Rótulos em cada ponto
for col, cor in [("Alta (0)", COR_ALTA), ("Óbito (1)", COR_OBITO)]:
    for i, dose in enumerate(doses):
        pct = ct_pct.loc[dose, col]
        n = ct.loc[dose, col]
        ax.text(
            i, pct + 1.8,
            f"{pct:.1f}%\n(n={int(n)})",
            ha="center", va="bottom",
            fontsize=12, color=TEXT
        )

# ── 6. EIXOS & GRID ──────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{d} dose{'s' if d != 1 else ''}" for d in doses],
    fontsize=15, color=TEXT
)
ax.set_ylabel("% within the dose group", fontsize=15, color=TEXT)
ax.set_xlabel("Number of vaccine doses", fontsize=12, color=TEXT)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.tick_params(colors=SUBTEXT, length=0)
ax.grid(axis="y", color=BORDER, linewidth=0.8, zorder=0)
for spine in ax.spines.values():
    spine.set_edgecolor(BORDER)

# ── 7. LEGENDA ──────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=COR_ALTA, edgecolor="#0F6E56", label="Discharge (0)"),
    mpatches.Patch(facecolor=COR_OBITO, edgecolor="#993C1D", label="Death (1)"),
]
ax.legend(handles=legend_elements, fontsize=12, frameon=True,
          edgecolor=BORDER, facecolor=BG, labelcolor=TEXT,
          loc="upper left")

# ── 8. TÍTULOS ──────────────────────────────────────────────────
ax.set_title("Outcome Distribution by Number of Vaccine Doses",
             fontsize=18, fontweight="bold", color=TEXT, pad=18)
fig.text(0.5, 0.01,
         "Percentage calculated within each dose group",
         ha="center", fontsize=15, color=SUBTEXT)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("grafico_vacinas_desfecho_area.png", dpi=180, bbox_inches="tight",
            facecolor=BG)
plt.show()
print("Salvo em grafico_vacinas_desfecho_area.png")
