import streamlit as st
import pandas as pd
from modelo import carregar_dados, treinar_modelos, prever

st.set_page_config(page_title="Análise Clínica Veterinária com IA", layout="centered")
st.title("🐾 Análise Clínica Veterinária com IA")
st.markdown("Insira a anamnese para prever o risco de eutanásia e cuidados clínicos.")

try:
    df, df_doencas = carregar_dados()
    features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
    features_eutanasia = features + ['tem_doenca_letal']

    modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias, le_mob, le_app, palavras_chave = treinar_modelos(
        df, features, features_eutanasia, df_doencas
    )
    modelos = (modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias)
except Exception as e:
    st.error(f"Erro ao carregar dados ou treinar modelos: {e}")
    st.stop()

texto = st.text_area("✍️ Digite a anamnese do paciente:")

if st.button("🔍 Analisar"):
    if not texto.strip():
        st.warning("⚠️ Digite a anamnese.")
    else:
        resultado = prever(texto, modelos, le_mob, le_app, palavras_chave, features, features_eutanasia)
        st.subheader("📋 Resultado da Análise")
        for k, v in resultado.items():
            st.markdown(f"**{k}**: {v}")
