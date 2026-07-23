import streamlit as st
from app import ingest_document, ask
import os

st.set_page_config(page_title="RAG PDF", layout="wide")

st.title("📄 Chat con tu PDF (RAG + Groq)")

# subir PDF
uploaded_file = st.file_uploader("Sube tu PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF cargado, procesando...")

    ingest_document("temp.pdf")

    st.success("Listo! Ahora puedes preguntar")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    question = st.chat_input("Haz tu pregunta")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                result = ask(question)
                answer = result["answer"].content

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })