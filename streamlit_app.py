import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langchain_groq import ChatGroq


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Chat PDF IA", layout="wide")
st.title("📄 Chat con PDF (RAG + Groq)")

# 🔐 API KEY desde Streamlit Cloud
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# =========================
# PROCESAR PDF (CACHEADO)
# =========================
@st.cache_resource
def process_pdf(file_path):

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(pages)

    embedding = FastEmbedEmbeddings()

    # 🔥 IMPORTANTE: persistencia para reutilizar
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embedding,
        persist_directory="./local_db"
    )

    return True  # 👈 NO retornamos vectordb


# =========================
# CREAR CHAIN (SIN PARAMS)
# =========================
@st.cache_resource
def get_chain():

    embedding = FastEmbedEmbeddings()

    vectordb = Chroma(
        persist_directory="./local_db",
        embedding_function=embedding
    )

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )

    prompt = PromptTemplate.from_template("""
Responde SOLO con base en el contexto.

Si no está en el documento, responde:
"No encontré esa información en el PDF"

Contexto:
{context}

Pregunta:
{input}

Respuesta:
""")

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Página {d.metadata.get('page', '?')}]\n{d.page_content}"
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
uploaded_file = st.file_uploader("📤 Sube tu PDF", type="pdf")

if uploaded_file:

    # guardar temporal
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF cargado")

    # 🔥 solo procesa (no devuelve vectordb)
    process_pdf("temp.pdf")

    # 🔥 chain sin parámetros
    chain = get_chain()

    # memoria chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Haz una pregunta sobre el PDF")

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
                    answer = "⚠️ Error procesando la respuesta"

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })