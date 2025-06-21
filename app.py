import streamlit as st
from modelo import carregar_dados, treinar_modelos, prever

st.set_page_config(page_title="Análise Clínica Veterinária", page_icon="🐾")
st.title("🐾 Análise Clínica Veterinária com IA")
st.markdown("Insira a anamnese para prever o risco de eutanásia e cuidados clínicos.")

# Carregar dados e treinar modelos
try:
    df, df_graves, df_comuns = carregar_dados()
    modelos, le_mob, le_app, palavras_chave_graves, palavras_chave_comuns, features, features_eutanasia = treinar_modelos(df, df_graves, df_comuns)
except Exception as e:
    st.error(f"Erro ao carregar dados ou treinar modelos: {e}")
    st.stop()

# Entrada do usuário
texto = st.text_area("📝 Digite a anamnese do paciente:")

if st.button("🔍 Analisar"):
    if not texto.strip():
        st.warning("Digite a anamnese.")
    else:
        resultado = prever(texto, modelos, le_mob, le_app, palavras_chave_graves, palavras_chave_comuns, features, features_eutanasia)
        st.subheader("📋 Resultado da Análise")
        for k, v in resultado.items():
            if isinstance(v, list):
                st.write(f"**{k}**:")
                for item in v:
                    st.markdown(f"- {item}")
            else:
                st.write(f"**{k}**: {v}")
