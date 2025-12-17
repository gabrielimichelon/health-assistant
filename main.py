import os
import json
import tempfile
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Áudio
from audio_recorder_streamlit import audio_recorder

# OpenAI (Whisper + Chat)
from openai import OpenAI

# LangChain + Qdrant
from typing import List, Optional
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


# -----------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# -----------------------------------------------------

load_dotenv()
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Assistente de Bem-Estar", page_icon="🌿")
st.title("Health Assistant — Seu especialista em bem-estar natural")


# -----------------------------------------------------
# MEMÓRIA PERSISTENTE
# -----------------------------------------------------

MEMORY_FILE = "memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def add_memory_entry(user_message):
    """Extrai sintomas e salva com a data."""
    llm_extract = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    extract_prompt = f"""
    Extraia sintomas mencionados no texto abaixo e retorne como lista JSON.

    Texto: "{user_message}"

    Responda apenas no formato:
    ["sintoma1", "sintoma2", ...]
    """

    try:
        extracted = llm_extract.invoke(extract_prompt).content
        symptoms = json.loads(extracted)
    except:
        symptoms = []

    if symptoms:
        memory = load_memory()
        memory.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "symptoms": symptoms,
            "text": user_message
        })
        save_memory(memory)


# -----------------------------------------------------
# RAG SETUP
# -----------------------------------------------------

# loader = TextLoader("instruct.txt")
# documents = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
# vectorstore = Qdrant.from_documents(
#     docs,
#     embeddings,
#     api_key=QDRANT_API_KEY,
#     url=QDRANT_URL,
#     collection_name=QDRANT_COLLECTION_NAME,
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado em bem-estar natural.

MEMÓRIA DO USUÁRIO (HISTÓRICO DE SINTOMAS):
{user_memory}

CONTEXTOS RECUPERADOS DO RAG:
{context}

INSTRUÇÕES:
- Utilize o histórico quando relevante.
- Utilize o contexto técnico do RAG quando necessário.
- Comente sobre recorrência de sintomas quando aplicável.
- Não invente informações.
- Se não houver dados suficientes, diga isso claramente.

Pergunta:
{question}
    """
)

def search_similar_documents(
    query: str,
    embedding_model: HuggingFaceEmbeddings,
    k: int = 3
) -> List[Document]:
    """
    Busca documentos similares no QDrant.
    
    Args:
        query: Texto de consulta
        embedding_model: Modelo de embedding (padrão: OpenAIEmbeddings)
        k: Número de resultados a retornar
    
    Returns:
        List[Document]: Lista de documentos similares
    """
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding_model
    )
    
    results = vectorstore.similarity_search(query, k=k)
    return results


def rag_pipeline(question):
    # memory = load_memory()

    # memory_text = "\n".join(
    #     [f"- {m['date']}: {m['text']} (sintomas: {', '.join(m['symptoms'])})"
    #      for m in memory]
    # )

    # docs = retriever.invoke(question)
    # context = "\n\n".join([doc.page_content for doc in docs])

    qdocs = search_similar_documents(query=question, embedding_model=embeddings)
    context = "\n\n".join([doc.page_content for doc in qdocs])

    memory_text=str(st.session_state.chat_history[-4:]) # últimas 2 conversas

    chain_input = {
        "context": context,
        "question": question,
        "user_memory": memory_text if memory_text else "Sem histórico armazenado."
    }

    response = llm.invoke(prompt.format(**chain_input))
    return response.content


# -----------------------------------------------------
# INTERFACE (TEXTO + ÁUDIO)
# -----------------------------------------------------

st.divider()

user_input = st.chat_input("Como posso ajudar hoje?")

audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000, text="",
    recording_color="#e8352c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="1x",)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------------------------------
# ENTRADA POR TEXTO
# -----------------------------------------------------

if user_input:
    add_memory_entry(user_input)

    with st.spinner("Analisando..."):
        resposta = rag_pipeline(user_input)

    st.session_state.chat_history.append(("Você", user_input))
    st.session_state.chat_history.append(("Assistente", resposta))


# -----------------------------------------------------
# ENTRADA POR ÁUDIO
# -----------------------------------------------------

if audio_bytes:
    st.success("Ouvindo...")

    # salvar temporário
    temp_audio = "temp_audio.wav"
    with open(temp_audio, "wb") as f:
        f.write(audio_bytes)

    with st.spinner(""):
        with open(temp_audio, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

    user_text = transcription.text
    st.write(f" **{user_text}**")

    add_memory_entry(user_text)

    with st.spinner("Analisando..."):
        resposta = rag_pipeline(user_text)

    st.session_state.chat_history.append(("Você (áudio)", user_text))
    st.session_state.chat_history.append(("Assistente", resposta))


# -----------------------------------------------------
# HISTÓRICO DO CHAT
# -----------------------------------------------------

for speaker, msg in st.session_state.chat_history:
    st.chat_message("user" if speaker.startswith("Você") else "assistant").markdown(msg)
