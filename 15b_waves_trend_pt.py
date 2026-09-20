import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from PIL import Image

# ── 1. CARREGAR DADOS ─────────────────────────────────────────────────────
df = pd.read_excel("703pacientes.xlsx")

# ── 2. PREPARAR VARIÁVEIS ─────────────────────────────────────────────────
df["Data de Entrada"] = pd.to_datetime(df["Data de Entrada"], dayfirst=True)

# ── 3. AGREGAR POR MÊS ────────────────────────────────────────────────────
mensal = df.groupby(df["Data de Entrada"].dt.to_period("M")).agg(
    n_internacoes=("Data de Entrada", "count"),
    n_obitos=("Óbito", "sum")
).reset_index()
mensal["n_altas"] = mensal["n_internacoes"] - mensal["n_obitos"]
mensal.rename(columns={"Data de Entrada": "mes"}, inplace=True)
mensal["mes_dt"] = mensal["mes"].dt.to_timestamp()

# ── 4. DEFINIR PERÍODOS DAS ONDAS ─────────────────────────────────────────
ondas = [
    {"label": "1ª Onda", "inicio": "2020-01-01", "fim": "2020-09-30", "cor": "#7b5ea7"},
    {"label": "2ª Onda", "inicio": "2020-10-01", "fim": "2021-05-31", "cor": "#c0823a"},
    {"label": "3ª Onda", "inicio": "2021-07-01", "fim": "2021-11-30", "cor": "#b03060"},
    {"label": "4ª Onda", "inicio": "2021-12-01", "fim": "2022-12-30", "cor": "#5b8db8"},
]

# ── 5. PLOTAR ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Fundo colorido por onda
for onda in ondas:
    ax.axvspan(
        pd.to_datetime(onda["inicio"]),
        pd.to_datetime(onda["fim"]),
        alpha=0.3, color=onda["cor"], label=onda["label"]
    )

# Area chart — internações
ax.fill_between(mensal["mes_dt"], mensal["n_internacoes"],
                 alpha=0.6, color="#2980b9", label="Internações")
ax.plot(mensal["mes_dt"], mensal["n_internacoes"],
        color="#2980b9", linewidth=2.8)

# Area chart — altas hospitalares
ax.fill_between(mensal["mes_dt"], mensal["n_altas"],
                 alpha=0.6, color="#27ae60", label="Altas")
ax.plot(mensal["mes_dt"], mensal["n_altas"],
        color="#1e8449", linewidth=2.8)

# Area chart — óbitos
ax.fill_between(mensal["mes_dt"], mensal["n_obitos"],
                 alpha=0.7, color="#e74c3c", label="Óbitos")
ax.plot(mensal["mes_dt"], mensal["n_obitos"],
        color="#c0392b", linewidth=2.8)

# Linha tracejada — início da vacinação
vacinacao = pd.to_datetime("2021-01-19")
ax.axvline(x=vacinacao, color="black", linestyle="--", linewidth=2.8)
ax.text(vacinacao + pd.Timedelta(days=5), ax.get_ylim()[1] * 0.99,
        "Início da Vacinação contra COVID-19\n(19/01/2021)",
        fontsize=12, color="black", va="top")

# ── 6. FORMATAÇÃO ────────────────────────────────────────────────────────
ax.set_xlim(pd.to_datetime("2020-01-01"), pd.to_datetime("2022-12-31"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
ax.set_xlabel("Período", fontsize=15, fontweight="bold")
ax.set_ylabel("Número de Pacientes", fontsize=15, fontweight="bold")
ax.set_title("Tendências Temporais de Internações e Óbitos ao Longo das Ondas de "
             "COVID-19\n(Jan/2020 – Dez/2022)",
             fontsize=22, fontweight="bold", pad=25)

# Legenda dentro do gráfico
legend_elementos = [
    Line2D([0], [0], color="#2980b9", lw=3, label="Internações (total)"),
    Line2D([0], [0], color="#27ae60", lw=3, label="Alta"),
    Line2D([0], [0], color="#e74c3c", lw=3, label="Óbito"),
    Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="Início da Vacinação"),
    plt.Rectangle((0, 0), 1, 1, fc="#7b5ea7", alpha=0.5, label="1ª Onda"),
    plt.Rectangle((0, 0), 1, 1, fc="#c0823a", alpha=0.5, label="2ª Onda"),
    plt.Rectangle((0, 0), 1, 1, fc="#b03060", alpha=0.5, label="3ª Onda"),
    plt.Rectangle((0, 0), 1, 1, fc="#5b8db8", alpha=0.5, label="4ª Onda"),
]
ax.legend(handles=legend_elementos, loc="upper right", fontsize=10,
          framealpha=0.9, bbox_to_anchor=(1.001, 1.01))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── 8. ANOTAÇÕES ESTATÍSTICAS POR ONDA ───────────────────────────────────
stats_ondas = [
    {"x": "2020-05-01", "label": "1ª Onda\n(ref.)"},
    {"x": "2020-11-15", "label": "2ª Onda\nOR=0,67 [0,45–0,98]\np=0,041"},
    {"x": "2021-09-01", "label": "3ª Onda\nOR=0,64 [0,40–1,03]\np=0,068"},
    {"x": "2022-04-01", "label": "4ª Onda\nOR=0,14 [0,09–0,23]\np<0,001"},
]
for s in stats_ondas:
    ax.text(
        pd.to_datetime(s["x"]), 0.85,
        s["label"],
        transform=ax.get_xaxis_transform(),
        fontsize=8, color="#333333",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                  alpha=0.59, linewidth=0.9)
    )

ax.text(
    0.13, 0.99,
    "χ²=72,88, gl=3, p<0,001",
    transform=ax.transAxes,
    fontsize=9, color="#555555",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
              alpha=0.99, linewidth=0.6)
)

plt.tight_layout()

# ── 7. SALVAR ────────────────────────────────────────────────────────────
plt.savefig("grafico_tendencia_covid_pt.png", dpi=300, bbox_inches="tight")
plt.savefig("grafico_tendencia_covid_pt.pdf", bbox_inches="tight")
# TIFF com compressão LZW (o TIFF sem compressão do matplotlib passa de 50MB)
Image.open("grafico_tendencia_covid_pt.png").save(
    "grafico_tendencia_covid_pt.tiff", dpi=(300, 300), compression="tiff_lzw")
plt.show()
print("Gráfico salvo como PNG, PDF e TIFF com sucesso.")
