import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Box plot de exames bioquímicos, trombo-hemostáticos e inflamatórios por
# desfecho (Alta vs. Óbito), mediana por paciente, apenas variáveis com
# diferença estatisticamente significativa (Mann-Whitney U, p<0,05).
# Fonte: ExamesB_filtradaa.xlsx (formato longo: 1 linha por resultado de
# exame), cruzado por COD_PACIENTE com 703pacientes.xlsx para confirmar o
# desfecho.
# ─────────────────────────────────────────────────────────────────────────
XLSX_EXAMES = "ExamesB_filtradaa.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "boxplot_exames_bioquimicos.png")
OUTPUT_TIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "boxplot_exames_bioquimicos.tiff")

BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_ALTA = "#1D9E75"
COR_OBITO = "#E0703A"
COR_JANELA = "#D9D9D9"
COR_JANELA_M = "#C8D9EC"
COR_JANELA_F = "#F3D3DE"
COR_JANELA_M_LINHA = "#6D93BE"
COR_JANELA_F_LINHA = "#C97A94"

# ════════════════════════════════════════════════════════════
# 1. DEFINIÇÃO DAS VARIÁVEIS (rótulo -> exames/descrições a combinar)
#    e da janela de referência geral (valores aproximados de laboratório
#    clínico adulto, usados apenas como faixa ilustrativa de referência).
# ════════════════════════════════════════════════════════════
# Referência: tupla (min, max) = janela única ("geral"); dict {"M": (min,
# max), "F": (min, max)} = janelas separadas por sexo (exibidas como duas
# faixas sobrepostas no gráfico), usado só onde o intervalo difere bastante
# entre homens e mulheres.
VARS = {
    # categoria: [(rótulo, [(EXAME, DESCRICAO), ...], ref, unidade)]
    "Função renal": [
        ("Creatinina", [("CREATININA", "VALOR")],
         {"M": (0.7, 1.3), "F": (0.6, 1.1)}, "mg/dL"),
        ("Ureia", [("UREIA", "VALOR")], (15, 45), "mg/dL"),
    ],
    "Perfil trombo-hemostático": [
        ("D-dímero", [("QUANTIFICACAO DE DIMEROS 'D' DE FIBRINA", "VALOR")],
         (0, 0.5), "µg/mL"),
        ("TTPA", [("TTPA (TEMPO TROMBOPL PARC/E ATIVADO)", "RATIO")],
         (0.8, 1.2), "ratio"),
        ("TP/INR", [("TP (TEMPO DE PROTROMBINA)", "INR")], (0.8, 1.2), ""),
    ],
    "Controle glicêmico": [
        ("HbA1c", [("HEMOGLOBINA GLICOSILADA", "Resultado")], (4.0, 5.6), "%"),
        ("Glicemia/jejum", [("GLICEMIA JEJUM", "VALOR"), ("GLICEMIA", "VALOR")],
         (70, 99), "mg/dL"),
    ],
    "Função hepática": [
        ("TGO", [("TGO / AST - TRANSAMINASE GLUTAMICA OXALACETICA", "VALOR")],
         (5, 40), "U/L"),
        ("TGP", [("TGP / ALT - ALANINA AMINOTRANSFERASE", "VALOR")],
         (7, 56), "U/L"),
        ("Bilirrubina", [("BILIRRUBINA TOTAL E FRACOES", "BILIRRUBINA TOTAL"),
                          ("BILIRRUBINAS", "BILIRRUBINA TOTAL"),
                          ("BILIRRUBINA TOTAL", "VALOR")], (0.2, 1.2), "mg/dL"),
        ("Fosfatase alcalina", [("FOSFATASE ALCALINA", "VALOR")],
         (44, 147), "U/L"),
        ("Gama GT", [("GAMA GT", "VALOR")],
         {"M": (8, 61), "F": (5, 36)}, "U/L"),
        ("Albumina", [("ALBUMINA - SANGUE", "VALOR")], (3.5, 5.0), "g/dL"),
    ],
    "Função cardíaca": [
        ("Troponina I", [("TROPONINA I", "VALOR"), ("TROPONINA I", "RESULTADO"),
                          ("TROPONINA I - ALTA SENSIBILIDADE", "VALOR")],
         (0, 0.04), "ng/mL"),
        ("CPK", [("CPK - CREATINA QUINASE", "VALOR"),
                 ("CREATINOFOSFOQUINASE (CPK)", "RESULTADO")],
         {"M": (39, 308), "F": (26, 192)}, "U/L"),
        ("CK-MB", [("CK-MB - CREATINE PHOSPHOKINASE FRACAO 2", "VALOR"),
                   ("CREATINOFOSFOQUINASE OU CPK-FRACAO-MB (CADA)", "RESULTADO")],
         (0, 25), "U/L"),
    ],
    "Lipidograma": [
        ("Colesterol total", [("COLESTEROL TOTAL", "VALOR")], (125, 200), "mg/dL"),
        ("LDL colesterol", [("LDL COLESTEROL", "VALOR")], (0, 130), "mg/dL"),
        ("HDL colesterol", [("HDL COLESTEROL", "VALOR")],
         {"M": (40, 60), "F": (50, 70)}, "mg/dL"),
    ],
    "Perfil inflamatório": [
        ("Ferritina", [("FERRITINA", "Resultado")],
         {"M": (24, 336), "F": (11, 307)}, "ng/mL"),
        ("Proteína C reativa", [("PROTEINA C REATIVA", "VALOR"),
                                 ("DOSAGEM DE PROTEINA C REATIVA", "VALOR")],
         (0, 5), "mg/L"),
        ("Lactato", [("LACTATO", "VALOR"), ("LACTATO - SANGUE", "VALOR")],
         (0.5, 2.2), "mmol/L"),
        ("LDH", [("LDH - LACTATO DESIDROGENASE", "VALOR")], (140, 280), "U/L"),
    ],
    "Eletrólitos": [
        ("Cálcio ionizado", [("CALCIO IONICO", "VALOR")], (1.12, 1.32), "mmol/L"),
        ("Sódio", [("SODIO", "VALOR")], (136, 145), "mEq/L"),
        ("Potássio", [("POTASSIO", "VALOR")], (3.5, 5.1), "mEq/L"),
        ("Magnésio", [("MAGNESIO (SANGUE)", "VALOR"), ("MAGNESIO", "VALOR")],
         (1.6, 2.6), "mg/dL"),
    ],
}


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


# ════════════════════════════════════════════════════════════
# 2. CARREGAR DADOS E CALCULAR A MEDIANA POR PACIENTE, POR VARIÁVEL
# ════════════════════════════════════════════════════════════
exames = pd.read_excel(XLSX_EXAMES)
exames["valor_num"] = exames["RESULTADO"].apply(parse_resultado)

obito_por_paciente = exames.groupby("COD_PACIENTE")["Obito"].first()

resultados_teste = {}   # rótulo -> dict com p, n_alta, n_obito, categoria, ref, unidade
dados_paciente = {}     # rótulo -> DataFrame [COD_PACIENTE, valor, Obito]

for categoria, variaveis in VARS.items():
    for rotulo, combos, ref, unidade in variaveis:
        mask = False
        for exame, descricao in combos:
            mask = mask | ((exames["EXAME"] == exame) &
                            (exames["DESCRICAO"] == descricao))
        sub = exames[mask].dropna(subset=["valor_num"])
        if sub.empty:
            continue
        mediana = sub.groupby("COD_PACIENTE")["valor_num"].median()
        obito = obito_por_paciente.reindex(mediana.index)
        df_var = pd.DataFrame({"valor": mediana, "Obito": obito}).dropna()

        alta = df_var.loc[df_var["Obito"] == 0, "valor"]
        obito_grp = df_var.loc[df_var["Obito"] == 1, "valor"]
        if len(alta) < 5 or len(obito_grp) < 5:
            continue

        stat, p = mannwhitneyu(alta, obito_grp, alternative="two-sided")
        resultados_teste[rotulo] = {
            "categoria": categoria, "p": p, "n_alta": len(alta),
            "n_obito": len(obito_grp), "ref": ref, "unidade": unidade,
        }
        dados_paciente[rotulo] = {"alta": alta.values, "obito": obito_grp.values}

# Mantém apenas variáveis com diferença estatisticamente significativa
significativas = {k: v for k, v in resultados_teste.items() if v["p"] < 0.05}

print("=" * 90)
print(f"Exames avaliados: {len(resultados_teste)} | Significativos (p<0,05): "
      f"{len(significativas)}")
for rotulo, info in significativas.items():
    print(f"{rotulo:22s} | {info['categoria']:28s} | n_alta={info['n_alta']:4d} "
          f"n_obito={info['n_obito']:4d} | p={info['p']:.4g}")
print("=" * 90)


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")




def limites_robustos(rotulo):
    """Calcula xlim robusto (ignora outliers extremos) e ajusta a janela
    de referência para caber dentro dele."""
    dados = dados_paciente[rotulo]
    combinado = np.concatenate([dados["alta"], dados["obito"]])
    ref = significativas[rotulo]["ref"]
    if isinstance(ref, dict):
        ref_min = min(v[0] for v in ref.values())
        ref_max = max(v[1] for v in ref.values())
    else:
        ref_min, ref_max = ref

    p2, p98 = np.percentile(combinado, [2, 98])
    q1, q3 = np.percentile(combinado, [25, 75])
    iqr = q3 - q1
    lo = min(p2, ref_min, q1 - 1.5 * iqr)
    hi = max(p98, ref_max, q3 + 1.5 * iqr)
    lo = max(lo, combinado.min(), 0 if combinado.min() >= 0 else lo)
    hi = min(hi, combinado.max())
    folga = (hi - lo) * 0.08 if hi > lo else 1.0
    return lo - folga, hi + folga


# ════════════════════════════════════════════════════════════
# 3. FIGURA — PAINÉIS POR CATEGORIA (2 COLUNAS DE CATEGORIAS),
#    ATÉ 3 VARIÁVEIS POR LINHA DENTRO DE CADA CATEGORIA
# ════════════════════════════════════════════════════════════
categorias_com_dados = [c for c in VARS if any(
    rotulo in significativas for rotulo, *_ in VARS[c])]

NCOLS_VAR = 3  # nº de variáveis lado a lado dentro de cada categoria


def n_linhas_categoria(categoria):
    n_vars = sum(1 for rotulo, *_ in VARS[categoria] if rotulo in significativas)
    return max(1, -(-n_vars // NCOLS_VAR))  # ceil


# Divide as categorias em 2 colunas (esquerda / direita), preservando a
# ordem, como no layout de referência.
col_esq = categorias_com_dados[0::2]
col_dir = categorias_com_dados[1::2]

# ── Dimensionamento em polegadas reais (evita disputa de espaço entre
#    gridspecs aninhados: 1 "unidade" de altura = uma quantidade fixa e
#    conhecida de polegadas, igual nas duas colunas) ────────────────────
POL_TITULO_CAT = 0.42
POL_LINHA = 2.35
POL_ESPACADOR = 0.45  # respiro extra após cada linha de variáveis
POL_TOPO = 1.90   # título + subtítulo + legenda (com folga p/ não sobrepor)
POL_RODAPE = 1.15  # nota de rodapé (+ respiro para a última linha)


def unidades_coluna(categorias):
    """Lista de alturas (em polegadas) de cada linha da grade da coluna:
    uma entrada POL_TITULO_CAT por categoria, POL_LINHA por linha de
    variáveis dela, e um espaçador após CADA linha de variáveis — inclusive
    a última de cada categoria — para que os rótulos "n=" (desenhados
    abaixo dos eixos) nunca fiquem escondidos atrás do título opaco da
    categoria seguinte."""
    alturas = []
    for c in categorias:
        alturas.append(POL_TITULO_CAT)
        n_linhas = n_linhas_categoria(c)
        for _ in range(n_linhas):
            alturas.append(POL_LINHA)
            alturas.append(POL_ESPACADOR)
    return alturas


alturas_esq = unidades_coluna(col_esq)
alturas_dir = unidades_coluna(col_dir)
altura_conteudo = max(sum(alturas_esq), sum(alturas_dir))

# Completa a coluna mais curta com um espaçador invisível, para que 1
# unidade de altura valha o mesmo número de polegadas nas duas colunas.
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
                                  height_ratios=alturas, hspace=0.35, wspace=0.38)

    linha_atual = 0
    for categoria in categorias:
        vars_sig = [v for v in VARS[categoria] if v[0] in significativas]

        ax_titulo = fig.add_subplot(gs_col[linha_atual, :])
        ax_titulo.axis("off")
        ax_titulo.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, transform=ax_titulo.transAxes,
            facecolor="#DCE6F1", edgecolor="none", zorder=0))
        ax_titulo.text(0.012, 0.5, categoria, transform=ax_titulo.transAxes,
                        ha="left", va="center", fontsize=15, fontweight="bold",
                        color=TEXT)
        linha_atual += 1

        n_linhas = n_linhas_categoria(categoria)
        for i, (rotulo, combos, ref, unidade) in enumerate(vars_sig):
            r, c = divmod(i, NCOLS_VAR)
            # cada linha de variáveis é seguida por uma linha-espaçador na
            # grade (inclusive a última da categoria)
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

    if isinstance(ref, dict):
        # Duas janelas de referência sobrepostas (masculina/feminina)
        for (r_min, r_max), cor_faixa, cor_linha in (
            (ref["M"], COR_JANELA_M, COR_JANELA_M_LINHA),
            (ref["F"], COR_JANELA_F, COR_JANELA_F_LINHA),
        ):
            ax.axvspan(max(r_min, xlim[0]), min(r_max, xlim[1]), color=cor_faixa,
                       alpha=0.55, zorder=0)
            for limite in (r_min, r_max):
                if xlim[0] < limite < xlim[1]:
                    ax.axvline(limite, color=cor_linha, linewidth=1.1,
                                linestyle="--", zorder=1)
    else:
        # Janela de referência única (faixa cinza + limites tracejados)
        ref_min, ref_max = ref
        ax.axvspan(max(ref_min, xlim[0]), min(ref_max, xlim[1]), color=COR_JANELA,
                   alpha=0.55, zorder=0)
        for limite in (ref_min, ref_max):
            if xlim[0] < limite < xlim[1]:
                ax.axvline(limite, color="#9AA3AC", linewidth=1.1, linestyle="--",
                            zorder=1)

    bp = ax.boxplot(
        [dados["obito"], dados["alta"]], vert=False, widths=0.55,
        patch_artist=True, showfliers=True, whis=1.5,
        medianprops=dict(color="white", linewidth=1.8),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                         markeredgecolor=COR_ALTA, alpha=0.5),
        zorder=3,
    )
    cores_caixa = [COR_OBITO, COR_ALTA]
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
    ax.set_yticklabels(["Óbito", "Alta"], fontsize=10.5, color=TEXT)
    for tick, cor in zip(ax.get_yticklabels(), [COR_OBITO, COR_ALTA]):
        tick.set_color(cor)
        tick.set_fontweight("bold")

    ax.set_xlim(*xlim)
    ax.tick_params(axis="x", labelsize=9, colors=SUBTEXT)

    unidade = info["unidade"]
    titulo_var = f"{rotulo} ({unidade})" if unidade else rotulo
    ax.set_title(f"{titulo_var}  —  {formata_p(info['p'])}", fontsize=12,
                 fontweight="bold", color=TEXT, pad=10)

    ax.text(0.0, -0.26, f"Alta n={info['n_alta']}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=COR_ALTA, style="italic")
    ax.text(1.0, -0.26, f"Óbito n={info['n_obito']}", transform=ax.transAxes,
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
          f"{len(significativas)} de {len(resultados_teste)} exames com "
          "diferença significativa (Mann-Whitney U, p<0,05)",
          ha="left", va="top", fontsize=12.5, color=SUBTEXT)

legend_elements = [
    mpatches.Patch(facecolor=COR_ALTA, edgecolor=COR_ALTA, label="Alta — box plot"),
    mpatches.Patch(facecolor=COR_OBITO, edgecolor=COR_OBITO, label="Óbito — box plot"),
    mpatches.Patch(facecolor=COR_JANELA, edgecolor="#9AA3AC",
                   label="Janela de referência (geral, adulto)"),
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
          "exame durante a internação | Eixo ajustado para excluir outliers "
          "extremos (percentil 2-98 / 1,5×IQR) | Teste: Mann-Whitney U",
          ha="left", va="bottom", fontsize=9.5, style="italic", color=SUBTEXT)
fig.text(0.04, 0.10 / fig_h,
          "Janela de referência: valores aproximados de laboratório clínico "
          "adulto; estratificada por sexo (masc./fem.) para Creatinina, Gama "
          "GT, CPK, HDL colesterol e Ferritina, janela única (geral) para as "
          "demais variáveis | Sexo inferido empiricamente a partir de "
          "marcadores laboratoriais sexo-dimórficos (Creatinina, CPK, "
          "Ferritina), pois a codificação não é documentada na base original",
          ha="left", va="bottom", fontsize=9.5, style="italic", color=SUBTEXT)

plt.savefig(OUTPUT_PNG, dpi=170, bbox_inches="tight", facecolor=BG)
_buf_png = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
plt.savefig(_buf_png, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_png).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_png)

print(f"Gráfico salvo em: {OUTPUT_PNG} e {OUTPUT_TIFF}")
plt.show()
