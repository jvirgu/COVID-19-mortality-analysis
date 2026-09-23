import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from PIL import Image
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Box plot de exames bioquímicos, trombo-hemostáticos e inflamatórios por
# desfecho (Alta vs. Óbito), mediana por paciente, apenas variáveis com
# diferença estatisticamente significativa (Mann-Whitney U, p<0,05).
# Fonte: ExamesB_coorte703_36variaveis_corrigido.xlsx (formato longo, já com
# nome de variável padronizado em VARIAVEL e a faixa de referência de
# laboratório reportada em VALOR_REFERENCIA, inclusive faixas
# sexo-específicas quando aplicável).
# ─────────────────────────────────────────────────────────────────────────
XLSX_EXAMES = "ExamesB_coorte703_36variaveis_corrigido.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "boxplot_exames_bioquimicos_v2.png")
OUTPUT_TIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "boxplot_exames_bioquimicos_v2.tiff")

BG = "#FFFFFF"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_ALTA = "#1D9E75"
COR_OBITO = "#E0703A"
COR_JANELA = "#D9D9D9"
COR_JANELA_M = "#C8D9EC"
COR_JANELA_F = "#F3D3DE"
COR_JANELA_M_LINHA = "#6D93BE"
COR_JANELA_F_LINHA = "#C97A94"

# Categoria de cada variável (usada apenas para agrupar os painéis)
CATEGORIAS = {
    "Função renal": ["Creatinina", "Ureia", "Ácido úrico", "RPCU"],
    "Perfil trombo-hemostático": ["D-dímero", "TTPA", "TP/INR", "Fibrinogênio"],
    "Controle glicêmico": ["HbA1c", "Glicemia/jejum"],
    "Função hepática": ["TGO", "TGP", "Bilirrubina", "Fosfatase alcalina",
                         "Gama GT", "Albumina"],
    "Função cardíaca": ["Troponina I", "CPK", "CK-MB"],
    "Lipidograma": ["Colesterol total", "LDL colesterol", "HDL colesterol",
                     "Triglicérides"],
    "Perfil inflamatório": ["Ferritina", "Proteína C reativa", "Lactato",
                             "LDH", "IL-6", "Procalcitonina", "LCR-Lactato"],
    "Eletrólitos": ["Cálcio ionizado", "Sódio", "Potássio", "Magnésio",
                     "Cloro", "Cálcio total"],
}
CATEGORIA_DA_VARIAVEL = {v: cat for cat, vs in CATEGORIAS.items() for v in vs}


def parse_resultado(valor):
    """Converte string de resultado laboratorial em float, quando possível."""
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip()
    s = re.sub(r"^[<>]\s*", "", s)  # ">10.0" / "<0.5" -> usa o valor limite
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_ref_parte(texto):
    """Extrai (min, max) de um trecho de referência tipo 'X a Y', 'até X'
    ou '> X'. max/min = None quando o lado é aberto."""
    t = texto.strip().lower()
    numeros = re.findall(r"[-+]?\d+(?:,\d+)?", t)
    numeros = [float(n.replace(",", ".")) for n in numeros]
    if not numeros:
        return None
    if "até" in t:
        return (0.0, numeros[0])
    if ">" in texto:
        return (numeros[0], None)
    if "<" in texto:
        return (0.0, numeros[0])
    if len(numeros) >= 2:
        return (numeros[0], numeros[1])
    return None


def parse_referencia(texto):
    """Converte o texto de VALOR_REFERENCIA em (min, max) ou em
    {"M": (min, max), "F": (min, max)}. Retorna None quando não disponível
    ou não interpretável."""
    if not isinstance(texto, str) or "não disponível" in texto.lower():
        return None
    if "|" in texto and ("M:" in texto or "F:" in texto):
        partes = {}
        for trecho in texto.split("|"):
            trecho = trecho.strip()
            if trecho.upper().startswith("M:"):
                r = parse_ref_parte(trecho.split(":", 1)[1])
                if r:
                    partes["M"] = r
            elif trecho.upper().startswith("F:"):
                r = parse_ref_parte(trecho.split(":", 1)[1])
                if r:
                    partes["F"] = r
        return partes if "M" in partes and "F" in partes else None
    return parse_ref_parte(texto)


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS E CORRIGIR O NOME DA VARIÁVEL "Hb1Ac" -> "HbA1c"
# ════════════════════════════════════════════════════════════
exames = pd.read_excel(XLSX_EXAMES)
exames["VARIAVEL"] = exames["VARIAVEL"].replace({"Hb1Ac": "HbA1c"})
exames["valor_num"] = exames["RESULTADO"].apply(parse_resultado)

obito_por_paciente = exames.groupby("COD_PACIENTE")["Obito"].first()

# ════════════════════════════════════════════════════════════
# 2. MEDIANA POR PACIENTE, TESTE DE MANN-WHITNEY E REFERÊNCIA DE LABORATÓRIO
# ════════════════════════════════════════════════════════════
resultados_teste = {}
dados_paciente = {}

for variavel, sub_var in exames.groupby("VARIAVEL"):
    sub = sub_var.dropna(subset=["valor_num"])
    if sub.empty or variavel not in CATEGORIA_DA_VARIAVEL:
        continue

    mediana = sub.groupby("COD_PACIENTE")["valor_num"].median()
    obito = obito_por_paciente.reindex(mediana.index)
    df_var = pd.DataFrame({"valor": mediana, "Obito": obito}).dropna()

    alta = df_var.loc[df_var["Obito"] == 0, "valor"]
    obito_grp = df_var.loc[df_var["Obito"] == 1, "valor"]
    if len(alta) < 5 or len(obito_grp) < 5:
        continue

    stat, p = mannwhitneyu(alta, obito_grp, alternative="two-sided")

    ref_bruta = sub_var["VALOR_REFERENCIA"].mode()
    ref = parse_referencia(ref_bruta.iloc[0]) if len(ref_bruta) else None
    unidade_match = re.search(r"([A-Za-zµ%/]+)\s*$", str(ref_bruta.iloc[0])) \
        if len(ref_bruta) else None

    resultados_teste[variavel] = {
        "categoria": CATEGORIA_DA_VARIAVEL[variavel], "p": p,
        "n_alta": len(alta), "n_obito": len(obito_grp), "ref": ref,
    }
    dados_paciente[variavel] = {"alta": alta.values, "obito": obito_grp.values}

significativas = {k: v for k, v in resultados_teste.items() if v["p"] < 0.05}

print("=" * 90)
print(f"Exames avaliados: {len(resultados_teste)} | Significativos (p<0,05): "
      f"{len(significativas)}")
for rotulo, info in significativas.items():
    print(f"{rotulo:22s} | {info['categoria']:28s} | n_alta={info['n_alta']:4d} "
          f"n_obito={info['n_obito']:4d} | p={info['p']:.4g}")
print("=" * 90)


def limites_robustos(rotulo, folga_mult=0.18):
    """xlim robusto (ignora outliers extremos), com folga maior para
    espaçar melhor os ticks e valorizar as diferenças entre os grupos."""
    dados = dados_paciente[rotulo]
    combinado = np.concatenate([dados["alta"], dados["obito"]])
    ref = significativas[rotulo]["ref"]

    p2, p98 = np.percentile(combinado, [2, 98])
    q1, q3 = np.percentile(combinado, [25, 75])
    iqr = q3 - q1
    lo_candidatos = [p2, q1 - 1.5 * iqr]
    hi_candidatos = [p98, q3 + 1.5 * iqr]
    if ref:
        if isinstance(ref, dict):
            for r_min, r_max in ref.values():
                if r_min is not None:
                    lo_candidatos.append(r_min)
                if r_max is not None:
                    hi_candidatos.append(r_max)
        else:
            r_min, r_max = ref
            if r_min is not None:
                lo_candidatos.append(r_min)
            if r_max is not None:
                hi_candidatos.append(r_max)

    lo = min(lo_candidatos)
    hi = max(hi_candidatos)
    lo = max(lo, combinado.min(), 0 if combinado.min() >= 0 else lo)
    hi = min(hi, combinado.max())
    folga = (hi - lo) * folga_mult if hi > lo else 1.0
    return lo - folga, hi + folga


# ════════════════════════════════════════════════════════════
# 3. FIGURA — PAINÉIS POR CATEGORIA (2 COLUNAS DE CATEGORIAS),
#    ATÉ 3 VARIÁVEIS POR LINHA DENTRO DE CADA CATEGORIA
# ════════════════════════════════════════════════════════════
categorias_com_dados = [c for c in CATEGORIAS if any(
    v in significativas for v in CATEGORIAS[c])]

NCOLS_VAR = 3


def n_linhas_categoria(categoria):
    n_vars = sum(1 for v in CATEGORIAS[categoria] if v in significativas)
    return max(1, -(-n_vars // NCOLS_VAR))


col_esq = categorias_com_dados[0::2]
col_dir = categorias_com_dados[1::2]

POL_TITULO_CAT = 0.42
POL_GAP_TITULO = 0.38  # respiro entre o título da categoria e a 1ª linha
POL_LINHA = 2.35
POL_ESPACADOR = 0.85
POL_TOPO = 1.90
POL_RODAPE = 1.15


def unidades_coluna(categorias):
    alturas = []
    for c in categorias:
        alturas.append(POL_TITULO_CAT)
        alturas.append(POL_GAP_TITULO)
        n_linhas = n_linhas_categoria(c)
        for _ in range(n_linhas):
            alturas.append(POL_LINHA)
            alturas.append(POL_ESPACADOR)
    return alturas


alturas_esq = unidades_coluna(col_esq)
alturas_dir = unidades_coluna(col_dir)
altura_conteudo = max(sum(alturas_esq), sum(alturas_dir))

if sum(alturas_esq) < altura_conteudo:
    alturas_esq.append(altura_conteudo - sum(alturas_esq))
if sum(alturas_dir) < altura_conteudo:
    alturas_dir.append(altura_conteudo - sum(alturas_dir))

fig_h = POL_TOPO + altura_conteudo + POL_RODAPE
fig_w = 21

fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

top_frac = 1 - POL_TOPO / fig_h
bottom_frac = POL_RODAPE / fig_h

gs_principal = fig.add_gridspec(1, 2, left=0.04, right=0.985,
                                 top=top_frac, bottom=bottom_frac, wspace=0.10)


def desenha_coluna(gs_slot, categorias, alturas):
    gs_col = gs_slot.subgridspec(len(alturas), NCOLS_VAR,
                                  height_ratios=alturas, hspace=0.55, wspace=0.38)

    linha_atual = 0
    for categoria in categorias:
        vars_sig = [v for v in CATEGORIAS[categoria] if v in significativas]

        ax_titulo = fig.add_subplot(gs_col[linha_atual, :])
        ax_titulo.axis("off")
        ax_titulo.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, transform=ax_titulo.transAxes,
            facecolor="#DCE6F1", edgecolor="none", zorder=0))
        ax_titulo.text(0.012, 0.5, categoria, transform=ax_titulo.transAxes,
                        ha="left", va="center", fontsize=15, fontweight="bold",
                        color=TEXT)
        linha_atual += 2  # título + respiro

        n_linhas = n_linhas_categoria(categoria)
        for i, rotulo in enumerate(vars_sig):
            r, c = divmod(i, NCOLS_VAR)
            linha_grade = linha_atual + r * 2
            ax = fig.add_subplot(gs_col[linha_grade, c])
            desenha_boxplot(ax, rotulo)
        linha_atual += n_linhas * 2


def desenha_boxplot(ax, rotulo):
    info = significativas[rotulo]
    dados = dados_paciente[rotulo]
    ref = info["ref"]
    xlim = limites_robustos(rotulo)

    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)

    def desenha_faixa(r_min, r_max, cor_faixa, cor_linha):
        esq = r_min if r_min is not None else xlim[0]
        dir_ = r_max if r_max is not None else xlim[1]
        ax.axvspan(max(esq, xlim[0]), min(dir_, xlim[1]), color=cor_faixa,
                   alpha=0.55, zorder=0)
        for limite in (r_min, r_max):
            if limite is not None and xlim[0] < limite < xlim[1]:
                ax.axvline(limite, color=cor_linha, linewidth=1.1,
                            linestyle="--", zorder=1)

    if isinstance(ref, dict):
        desenha_faixa(*ref["M"], COR_JANELA_M, COR_JANELA_M_LINHA)
        desenha_faixa(*ref["F"], COR_JANELA_F, COR_JANELA_F_LINHA)
    elif ref is not None:
        desenha_faixa(ref[0], ref[1], COR_JANELA, "#9AA3AC")

    # Óbito em cima, Alta embaixo
    bp = ax.boxplot(
        [dados["alta"], dados["obito"]], vert=False, widths=0.55,
        patch_artist=True, showfliers=True, whis=1.5,
        medianprops=dict(color="white", linewidth=1.8),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                         markeredgecolor=COR_ALTA, alpha=0.5),
        zorder=3,
    )
    cores_caixa = [COR_ALTA, COR_OBITO]
    for patch, cor in zip(bp["boxes"], cores_caixa):
        patch.set_facecolor(cor)
        patch.set_edgecolor(cor)
        patch.set_alpha(0.9)
    for elemento in ("whiskers", "caps"):
        for linha, cor in zip(np.array(bp[elemento]).reshape(2, -1),
                               cores_caixa):
            for l in np.atleast_1d(linha):
                l.set_color(cor)

    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Alta", "Óbito"], fontsize=10.5, color=TEXT)
    for tick, cor in zip(ax.get_yticklabels(), [COR_ALTA, COR_OBITO]):
        tick.set_color(cor)
        tick.set_fontweight("bold")

    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=4))
    ax.tick_params(axis="x", labelsize=9, colors=SUBTEXT)

    ax.text(0.5, 1.22, formata_p(info["p"]), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10.5, color=SUBTEXT)
    ax.set_title(rotulo, fontsize=13, fontweight="bold", color=TEXT, pad=16)

    ax.text(0.0, -0.24, f"Alta n={info['n_alta']}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=COR_ALTA, style="italic")
    ax.text(1.0, -0.24, f"Óbito n={info['n_obito']}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=COR_OBITO, style="italic")


desenha_coluna(gs_principal[0, 0], col_esq, alturas_esq)
desenha_coluna(gs_principal[0, 1], col_dir, alturas_dir)

# ── Título, subtítulo e legenda ─────────────────────────────────────────
y_titulo = 1 - (0.42 / fig_h)
y_subtitulo = 1 - (0.85 / fig_h)
y_legenda = 1 - (1.38 / fig_h)

fig.text(0.04, y_titulo,
          "Exames Bioquímicos, Trombo-Hemostáticos e Inflamatórios por "
          "Desfecho — Mediana por Paciente — Box Plot — Variáveis "
          "Significativas",
          ha="left", va="top", fontsize=19, fontweight="bold", color=TEXT)
fig.text(0.04, y_subtitulo,
          "Box plot por paciente, por Alta vs. Óbito hospitalar | "
          "HCFMRP-USP",
          ha="left", va="top", fontsize=12.5, color=SUBTEXT)

legend_elements = [
    mpatches.Patch(facecolor=COR_ALTA, edgecolor=COR_ALTA, label="Alta — box plot"),
    mpatches.Patch(facecolor=COR_OBITO, edgecolor=COR_OBITO, label="Óbito — box plot"),
    mpatches.Patch(facecolor=COR_JANELA, edgecolor="#9AA3AC",
                   label="Janela de referência (geral)"),
    mpatches.Patch(facecolor=COR_JANELA_M, edgecolor=COR_JANELA_M_LINHA,
                   label="Janela de referência (masc.)"),
    mpatches.Patch(facecolor=COR_JANELA_F, edgecolor=COR_JANELA_F_LINHA,
                   label="Janela de referência (fem.)"),
]
fig.legend(handles=legend_elements, loc="upper left",
           bbox_to_anchor=(0.04, y_legenda), fontsize=11, frameon=False,
           labelcolor=TEXT, ncol=5, columnspacing=1.5, handlelength=1.4)

fig.text(0.04, 0.34 / fig_h,
          "Mediana por paciente calculada a partir de todos os resultados do "
          "exame durante a internação | Eixos com maior espaçamento entre "
          "marcações para evidenciar a diferença entre os grupos | "
          "Teste: Mann-Whitney U",
          ha="left", va="bottom", fontsize=9.5, style="italic", color=SUBTEXT)
fig.text(0.04, 0.10 / fig_h,
          "Janela de referência: valores de laboratório clínico reportados "
          "no próprio resultado do exame (VALOR_REFERENCIA), estratificados "
          "por sexo quando disponível | Sexo M/F conforme cadastro do "
          "laboratório",
          ha="left", va="bottom", fontsize=9.5, style="italic", color=SUBTEXT)

plt.savefig(OUTPUT_PNG, dpi=170, bbox_inches="tight", facecolor=BG)
_buf_png = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
plt.savefig(_buf_png, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_png).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_png)

print(f"Gráfico salvo em: {OUTPUT_PNG} e {OUTPUT_TIFF}")
plt.show()
