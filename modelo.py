import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def normalizar_texto(texto):
    texto = unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extrair_variavel(padrao, texto, tipo=float, valor_padrao=None):
    match = re.search(padrao, texto)
    if match:
        try:
            return tipo(match.group(1).replace(',', '.'))
        except:
            return valor_padrao
    return valor_padrao

def carregar_dados():
    df = pd.read_csv("data/casos_clinicos.csv")
    df_doencas_graves = pd.read_csv("data/doencas_caninas_eutanasia_expandidas.csv")
    df_doencas_comuns = pd.read_csv("data/top100_doencas_caninas.csv")

    return df, df_graves, df_comuns

def treinar_modelos(df, features, features_eutanasia, df_graves, df_comuns):
    le_mob = LabelEncoder()
    le_app = LabelEncoder()

    df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].astype(str).str.lower().str.strip())
    df['Apetite'] = le_app.fit_transform(df['Apetite'].astype(str).str.lower().str.strip())

    palavras_graves = [normalizar_texto(d) for d in df_graves['Doenca'].dropna().unique()]
    palavras_comuns = [normalizar_texto(d) for d in df_comuns['Doenca'].dropna().unique()]

    df['Doenca'] = df['Doenca'].fillna('').apply(normalizar_texto)
    df['tem_doenca_letal'] = df['Doenca'].apply(lambda d: int(any(p in d for p in palavras_graves)))

    X_e = df[features_eutanasia]
    y_e = df['Eutanasia']

    modelo_eutanasia = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_eutanasia.fit(X_e, y_e)

    modelo_alta = RandomForestClassifier().fit(df[features], df['Alta'])
    modelo_internar = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_internar.fit(df[features], df['Internar'])

    modelo_dias = RandomForestRegressor().fit(
        df[df['Internar'] == 1][features],
        df[df['Internar'] == 1]['Dias Internado']
    )

    return (modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias), le_mob, le_app, palavras_graves, palavras_comuns

def prever(anamnese, modelos, le_mob, le_app, palavras_graves, palavras_comuns, features, features_eutanasia):
    modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias = modelos
    texto_norm = normalizar_texto(anamnese)

    idade = extrair_variavel(r"(\d+(?:\.\d+)?)\s*anos?", texto_norm, float, 5.0)
    peso = extrair_variavel(r"(\d+(?:\.\d+)?)\s*kg", texto_norm, float, 10.0)
    temperatura = extrair_variavel(r"(\d{2}(?:[.,]\d+)?)\s*(?:graus|c|celsius|\u00b0c)", texto_norm, float, 38.5)
    gravidade = 10 if "vermelho" in texto_norm else 5

    if any(p in texto_norm for p in ["dor intensa", "dor severa", "dor forte"]):
        dor = 10
    elif "dor moderada" in texto_norm:
        dor = 5
    elif any(p in texto_norm for p in ["sem dor", "ausencia de dor"]):
        dor = 0
    else:
        dor = 4

    if any(p in texto_norm for p in ["sem apetite", "nao come", "perda de apetite"]):
        apetite = le_app.transform(["nenhum"])[0] if "nenhum" in le_app.classes_ else 0
    elif any(p in texto_norm for p in ["baixo apetite", "apetite baixo"]):
        apetite = le_app.transform(["baixo"])[0] if "baixo" in le_app.classes_ else 0
    else:
        apetite = le_app.transform(["normal"])[0] if "normal" in le_app.classes_ else 0

    if any(p in texto_norm for p in ["nao anda", "sem andar", "incapaz de andar"]):
        mobilidade = le_mob.transform(["sem andar"])[0] if "sem andar" in le_mob.classes_ else 0
    elif any(p in texto_norm for p in ["mobilidade limitada", "dificuldade locomotora"]):
        mobilidade = le_mob.transform(["limitada"])[0] if "limitada" in le_mob.classes_ else 0
    else:
        mobilidade = le_mob.transform(["normal"])[0] if "normal" in le_mob.classes_ else 0

    doencas_graves_detectadas = [d for d in palavras_graves if d in texto_norm]
    doencas_comuns_detectadas = [d for d in palavras_comuns if d in texto_norm]

    tem_doenca_letal = int(len(doencas_graves_detectadas) > 0)

    entrada = pd.DataFrame([[idade, peso, gravidade, dor, mobilidade, apetite, temperatura, tem_doenca_letal]],
                           columns=features_eutanasia)

    prob_eutanasia = modelo_eutanasia.predict_proba(entrada)[0][1]

    if tem_doenca_letal:
        prob_eutanasia += 0.35

    prob_eutanasia = min(prob_eutanasia, 1.0)

    internar = 1 if prob_eutanasia > 0.4 else 0
    alta = 0 if internar or prob_eutanasia > 0.5 else 1

    dias = 0
    if internar:
        dias = int(round(modelo_dias.predict(entrada[features])[0]))
        dias = max(dias, 2)

    return {
        "Alta": "Sim" if alta else "Não",
        "Internar": "Sim" if internar else "Não",
        "Dias Internado": dias,
        "Chance de Eutanásia (%)": round(prob_eutanasia * 100, 1),
        "Doenças Detectadas": doencas_comuns_detectadas or ["Nenhuma identificada"],
        "Doenças Graves Detectadas": doencas_graves_detectadas or ["Nenhuma grave"]
    }
