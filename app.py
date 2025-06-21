import streamlit as st
import pandas as pd
from modelo import carregar_dados, treinar_modelos, prever

st.title("🐾 Análise Clínica Veterinária com IA")
st.markdown("Insira a anamnese para prever o risco de eutanásia e cuidados clínicos.")

try:
    df, df_doencas_graves, df_doencas_comuns = carregar_dados()
    modelos, le_mob, le_app, palavras_chave = treinar_modelos(df, df_doencas_graves, df_doencas_comuns)
except Exception as e:
    st.error(f"Erro ao carregar dados ou treinar modelos: {e}")
    st.stop()

texto = st.text_area("✍️ Digite a anamnese do paciente:")

if st.button("🔍 Analisar"):
    if not texto.strip():
        st.warning("Digite a anamnese.")
    else:
        resultado = prever(texto, modelos, le_mob, le_app, palavras_chave)
        st.subheader("📋 Resultado da Análise")
        for k, v in resultado.items():
            st.write(f"**{k}**: {v}")


