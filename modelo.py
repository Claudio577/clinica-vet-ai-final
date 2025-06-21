import pandas as pd
import re
import unicodedata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def normalizar_texto(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    return texto

def carregar_dados():
    df = pd.read_csv("dados/casos_clinicos.csv")
    df_doencas = pd.read_csv("dados/doencas_graves.csv")
    return df, df_doencas

def treinar_modelos(df, df_doencas):
    le_mob = LabelEncoder()
    le_app = LabelEncoder()

    df["Mobilidade"] = le_mob.fit_transform(df["Mobilidade"].str.lower())
    df["Apetite"] = le_app.fit_transform(df["Apetite"].str.lower())

    palavras_chave = df_doencas["Doença"].str.lower().tolist()

    df["tem_doenca_letal"] = df["Doença"].apply(lambda d: int(any(p in d.lower() for p in palavras_chave)))

    features = ["Idade", "Peso", "Gravidade", "Dor", "Mobilidade", "Apetite", "Temperatura"]
    features_e = features + ["tem_doenca_letal"]

    X = df[features_e]
    y_eutanasia = df["Eutanasia"]
    y_alta = df["Alta"]
    y_internar = df["Internar"]
    y_dias = df["Dias Internado"]

    modelo_eutanasia = RandomForestClassifier().fit(X, y_eutanasia)
    modelo_alta = RandomForestClassifier().fit(X, y_alta)
    modelo_internar = RandomForestClassifier().fit(X, y_internar)
    modelo_dias = RandomForestRegressor().fit(X[y_internar == 1], y_dias[y_internar == 1])

    return (modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias), le_mob, le_app, palavras_chave

def prever(anamnese, modelos, le_mob, le_app, palavras_chave):
    texto = normalizar_texto(anamnese)
    idade, peso, temp, gravidade, dor = 8, 15, 38.8, 5, 4
    apetite = le_app.transform(["baixo"])[0] if "baixo" in le_app.classes_ else 0
    mobilidade = le_mob.transform(["limitada"])[0] if "limitada" in le_mob.classes_ else 0
    tem_doenca_letal = int(any(p in texto for p in palavras_chave))

    entrada = pd.DataFrame([[idade, peso, gravidade, dor, mobilidade, apetite, temp, tem_doenca_letal]],
                           columns=["Idade", "Peso", "Gravidade", "Dor", "Mobilidade", "Apetite", "Temperatura", "tem_doenca_letal"])

    modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias = modelos

    prob_eutanasia = modelo_eutanasia.predict_proba(entrada)[0][1]
    alta = modelo_alta.predict(entrada)[0]
    internar = modelo_internar.predict(entrada)[0]
    dias = int(round(modelo_dias.predict(entrada)[0])) if internar else 0

    doencas = [p for p in palavras_chave if p in texto]

    return {
        "Alta": "Sim" if alta else "Não",
        "Internar": "Sim" if internar else "Não",
        "Dias Internado": dias,
        "Chance de Eutanásia (%)": round(prob_eutanasia * 100, 1),
        "Doenças Detectadas": doencas if doencas else ["Nenhuma grave"]
    }
