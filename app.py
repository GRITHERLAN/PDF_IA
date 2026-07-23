from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langchain_groq import ChatGroq
import os


def ingest_document(pdf_path):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    embedding = FastEmbedEmbeddings()

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embedding,
        persist_directory="./local_db"
    )

    return vectordb


def create_chain():

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate.from_template("""
Responde SOLO con base en el contexto.

Contexto:
{context}

Pregunta:
{input}

Respuesta:
""")

    embedding = FastEmbedEmbeddings()

    vectordb = Chroma(
        persist_directory="./local_db",
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(
            f"[Página {d.metadata.get('page')}]\n{d.page_content}"
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


def ask(question):
    chain = create_chain()
    result = chain.invoke(question)
    return result