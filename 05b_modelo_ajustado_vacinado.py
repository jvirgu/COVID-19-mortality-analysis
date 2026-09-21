"""
Modelo de regressão logística ajustado (multivariável) para Óbito,
incluindo Vacinado como uma das covariáveis — versão adaptada do
modelo em 05_statistical_analysis_logisticregression.py (linhas
274-306) para o schema atual de New_pacientes703.xlsx.

Adaptações em relação ao script 05 original:
- Fonte de dados: New_pacientes703.xlsx (script 05 usa 703pacientes.xlsx,
  que tinha colunas de fabricante de vacina e dummies já prontas que não
  existem nesta versão da base).
- Idade_cat, Estado_Civil e Grau de Instrução são recodificadas aqui em
  dummies (categoria de referência = 0), pois chegam como uma única
  coluna categórica nesta base.
- Prob_Infec excluída: é constante (=1 para todos os pacientes, sem
  variância) e não pode entrar no modelo.
- UF excluída: separação quase perfeita (apenas 2 pacientes com UF=1,
  nenhum óbito), o que gera coeficiente instável (|β|>19, IC infinito).
- Variáveis de fabricante de vacina (Pfizer, Aztrazeneca, Janssen,
  Coronavac_Butantan, Nenhuma_dose) e Trombose/Sepse/Choque (separados)
  não estão presentes nesta base e foram omitidas.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statstests.process import stepwise

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "New_pacientes703.xlsx")

dados = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

dados["Idade_cat_1"] = (dados["Idade_cat"] == 1).astype(int)
dados["Idade_cat_2"] = (dados["Idade_cat"] == 2).astype(int)
dados["Idade_cat_3"] = (dados["Idade_cat"] == 3).astype(int)
dados["Estado_Civil_1"] = (dados["Estado_Civil"] == 1).astype(int)
dados["Estado_Civil_2"] = (dados["Estado_Civil"] == 2).astype(int)
dados["Grau_Instrucao_1"] = (dados["Grau de Instrução"] == 1).astype(float)
dados["Grau_Instrucao_2"] = (dados["Grau de Instrução"] == 2).astype(float)
dados.loc[dados["Grau de Instrução"].isna(), ["Grau_Instrucao_1", "Grau_Instrucao_2"]] = np.nan

formula = (
    "Óbito ~ Vacinado + Sexo + Prob_Card + CP + Município + Diabetes + "
    "SRAG + Choques + Prob_neurol + Prob_Hemat + Cancer + Prob_Resp + "
    "Prob_Metab + Prob_TGI + Prob_Hep + Prob_Hid_Elet + Prob_AI_Infla + "
    "Febre + Outros + Traumatismo + COVID_CRÍTICA + Prob_Renal + LRA + "
    "Dias_permanência + Estado_Civil_1 + Estado_Civil_2 + Idade_cat_1 + "
    "Idade_cat_2 + Idade_cat_3 + Grau_Instrucao_1 + Grau_Instrucao_2"
)

modelo_rl = smf.glm(formula=formula, data=dados, family=sm.families.Binomial()).fit()
print(modelo_rl.summary())
print("\nN usado no modelo (listwise deletion):", int(modelo_rl.nobs))

or_values = np.exp(modelo_rl.params)
conf = modelo_rl.conf_int()
conf["OR_inf"] = np.exp(conf[0])
conf["OR_sup"] = np.exp(conf[1])
resultado = pd.DataFrame({
    "Coef (β)": modelo_rl.params,
    "OR": or_values,
    "IC_inf": conf["OR_inf"],
    "IC_sup": conf["OR_sup"],
    "p-value": modelo_rl.pvalues,
})
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 160)
print("\n=== Modelo completo (ajustado) ===")
print(resultado.sort_values("p-value").to_string())

print("\n=== Seleção Stepwise (p < 0.05) ===")
step_modelo = stepwise(modelo_rl, pvalue_limit=0.05)
