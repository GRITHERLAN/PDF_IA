import streamlit as st
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

from langchain_groq import ChatGroq


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Chat IA", layout="wide")
st.title("Chat con PDF o CSV")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# =========================
# PROCESAR PDF
# =========================
def process_pdf(file_bytes):

    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader("temp.pdf")

    pages = loader.load()[:15]  # limitar páginas

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(pages)

    return chunks


# =========================
# PROCESAR CSV
# =========================
def process_csv(file):

    df = pd.read_csv(file)

    # 🔥 limitar filas para no explotar memoria
    df = df.head(200)

    documents = []

    for i, row in df.iterrows():
        content = ", ".join([f"{col}: {row[col]}" for col in df.columns])

        documents.append(
            Document(
                page_content=content,
                metadata={"row": i}
            )
        )

    return documents


# =========================
# CREAR VECTOR DB
# =========================
def create_vectordb(documents):

    embedding = FastEmbedEmbeddings()

    vectordb = Chroma.from_documents(
        documents,
        embedding=embedding
    )

    return vectordb


# =========================
# CREAR CHAIN
# =========================
def get_chain(vectordb):

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )

    prompt = PromptTemplate.from_template("""
Responde SOLO con base en el contexto.
tambien responde de que trata el pdf o libro compartido.
Usa TODA la información relevante del contexto.
Si hay múltiples fragmentos, combínalos.
Sé claro, preciso y completo.
Si no encuentras la respuesta, dilo claramente.
Cita la fuente cuando sea posible.
limitate a hablar solo del archivo que te pasen.
si te preguntan algo que no tiene que ver con el pdf, responde que no fuiste entrenado para ello.

Si no está en el documento, responde:
"No encontré esa información en el archivo"

Contexto:
{context}

Pregunta:
{input}

Respuesta:
""")

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 2}
    )

    def format_docs(docs):
        return "\n\n".join(
            f"{d.page_content}"
            for d in docs
        )

    chain = RunnableParallel({
        "context": retriever,
        "input": RunnablePassthrough()
    }) | RunnablePassthrough.assign(
        answer=(
            {
                "context": lambda x: format_docs(x["context"]),
                "input": lambda x: x["input"]
            }
            | prompt
            | llm
        )
    )

    return chain


# =========================
# UI
# =========================
uploaded_file = st.file_uploader(
    "Sube tu archivo (PDF o CSV)",
    type=["pdf", "csv"]
)

if uploaded_file:

    # 🔥 evitar reprocesar cada vez
    if "vectordb" not in st.session_state:

        with st.spinner("Procesando archivo..."):

            if uploaded_file.type == "application/pdf":
                documents = process_pdf(uploaded_file.read())

            elif uploaded_file.type == "text/csv":
                documents = process_csv(uploaded_file)

            else:
                st.error("Formato no soportado")
                st.stop()

            vectordb = create_vectordb(documents)

            st.session_state.vectordb = vectordb
            st.session_state.chain = get_chain(vectordb)

    chain = st.session_state.chain

    # memoria del chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Haz una pregunta sobre el archivo")

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    result = chain.invoke(question)
                    answer = result["answer"].content
                except Exception as e:
                    st.error(e)
                    answer = str(e)

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })