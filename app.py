from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# =========================
# 📥 INGESTA MEJORADA
# =========================
def ingest_document(pdf_path):
    print("📄 Cargando documento...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 🔥 Mejor chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(pages)

    print(f"Convertimos {len(pages)} páginas en {len(chunks)} chunks")

    # 🔥 Embeddings más robustos
    embedding = FastEmbedEmbeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./local_knowledge_db"
    )

    print("✅ Documento indexado correctamente")

    return vector_store


# =========================
# 🧠 RAG MEJORADO
# =========================
def create_local_rag_chain():

    # 🔥 Mejor modelo (si tienes RAM suficiente)
    llm = ChatOllama(
        model="mistral",
        temperature=0.3  # más preciso
    )

    #  Prompt mejorado
    prompt = PromptTemplate.from_template("""
Eres un asistente experto en análisis de documentos.

INSTRUCCIONES:
- Usa TODA la información relevante del contexto
- Si hay múltiples fragmentos, combínalos
- Sé claro, preciso y completo
- Si no encuentras la respuesta, dilo claramente
- Cita la fuente cuando sea posible
- limitate a hablar solo del archivo que te pasen
- si te preguntan algo que no tiene que ver con el pdf, responde que no fuiste entrenado para ello                                         
                                          
Contexto:
{context}

Pregunta:
{input}

Respuesta detallada:
""")

    embedding = FastEmbedEmbeddings()

    vector_store = Chroma(
        persist_directory="./local_knowledge_db",
        embedding_function=embedding
    )

    # 🔥 Mejor retrieval (SIN limitar demasiado)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}
    )

    # Formatear documentos
    def format_docs(docs):
        return "\n\n".join(
            f"[Página {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    # 🔥 Chain con respuesta + fuentes
    rag_chain = RunnableParallel(
        {
            "context": retriever,
            "input": RunnablePassthrough(),
        }
    ) | RunnablePassthrough.assign(
        answer=(
            {
                "context": lambda x: format_docs(x["context"]),
                "input": lambda x: x["input"],
            }
            | prompt
            | llm
        )
    )

    return rag_chain


# =========================
# 💬 CHAT
# =========================
def chat_with_document(question):
    chain = create_local_rag_chain()

    print(f"\n🤔 Pregunta: {question}")
    print("🤖 Buscando en los documentos...")

    result = chain.invoke(question)

    # 🔥 DEBUG (muy importante)
    print(f"\n📊 DEBUG: {len(result['context'])} fragmentos recuperados")

    # Respuesta
    print(f"\n💡 Respuesta:\n{result['answer'].content}")

    # Fuentes
    print("\n📚 Fuentes consultadas:")

    for i, doc in enumerate(result["context"], 1):
        print(f"{i}. Página {doc.metadata.get('page', '?')}")

    return result


# =========================
# 🚀 MAIN
# =========================
def main():
    print("🚀 Iniciando tu IA local con RAG...")

    pdf_path = "libro.pdf"
    ingest_document(pdf_path)

    print("\n💬 Ya puedes conversar con tu documento")
    print("Escribe 'salir' para terminar")

    while True:
        question = input("\n📝 Tu pregunta: ")

        if question.lower() == 'salir':
            break

        chat_with_document(question)


if __name__ == "__main__":
    main()