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
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore


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

st.caption("É possível melhorar sua qualidade de vida com recursos naturais e hábitos saudáveis. Estou aqui para te ajudar nesse processo contribuindo para uma vida mais saudável!  ")

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

def add_memory_entry(user_message) -> str:
    """Extrai sintomas e salva com a data.

    Args:
        query: Texto de consulta
    
    Returns:
        Str: Os sintomas extraídos do texto ou o texto de consulta se não for encontrado sintoma
    """
    llm_extract = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    extract_prompt = f"""
    Extraia sintomas mencionados no texto abaixo e retorne como lista JSON.

    Texto: "{user_message}"

    Responda apenas no formato:
    ["sintoma1", "sintoma2", ...]
    """

    main_content = user_message
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
        if isinstance(symptoms, list):
            main_content = ", ".join([m for m in symptoms])
    
    return main_content


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
- UTILIZE EXCLUSIVAMENTE OS CONTEXTOS RECUPERADOS DO RAG PARA CRIAR AS RESPOSTAS. 
- Comente sobre recorrência de sintomas quando aplicável.
- Não invente informações.
- Se não houver dados suficientes, diga isso claramente.
- Preciso de um texto com no máximo 1000 caracteres.

Pergunta:
{question}
    """
)

def search_similar_documents(
    query: str,
    embedding_model: HuggingFaceEmbeddings,
    k: int = 5
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

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embedding_model,
    )
    
    results= vector_store.similarity_search_with_relevance_scores(query=query, k=k)
    return results


def rag_pipeline(question, main_content):
    memory = load_memory()

    memory_text = "\n".join(
        [f"- {m['date']}: {m['text']} (sintomas: {', '.join(m['symptoms'])})"
         for m in memory[-2:]]
    )
    # memory_text=str(st.session_state.chat_history[-4:]) # últimas 2 conversas

    # Se a pergunta do usuário não fornece um sintoma, utiliza o sintoma da última pergunta para auxiliar o RAG
    if question == main_content and len(memory) > 0 and memory[-1]["text"] != question:
        aux_conent = ", ".join([m for m in memory[-1]['symptoms']])
        main_content = f"{aux_conent}, {question}"

    qdocs = search_similar_documents(query=main_content, embedding_model=embeddings)
    context = "\n\n".join([doc[0].page_content for doc in qdocs])

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

# st.divider()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create a placeholder
placeholder = st.empty()
 
with st.container():
    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000, text="",
        recording_color="#e8352c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="1x",)

    user_input = st.chat_input("Como posso ajudar hoje?") 


# -----------------------------------------------------
# ENTRADA POR TEXTO
# -----------------------------------------------------

if user_input:
    with placeholder.container(height=550):
        if st.session_state.chat_history:
            for speaker, msg in st.session_state.chat_history:
                st.chat_message("user" if speaker.startswith("Você") else "assistant", avatar= "🤷" if speaker.startswith("Você") else "👩‍🌾").markdown(msg)

    with st.spinner("Analisando..."):
        main_content = add_memory_entry(user_input)
        resposta = rag_pipeline(user_input, main_content)

    st.session_state.chat_history.append(("Você", user_input))
    st.session_state.chat_history.append(("Assistente", resposta))


# -----------------------------------------------------
# ENTRADA POR ÁUDIO
# -----------------------------------------------------

if audio_bytes:
    with placeholder.container(height=550):
        if st.session_state.chat_history:
            for speaker, msg in st.session_state.chat_history:
                st.chat_message("user" if speaker.startswith("Você") else "assistant", avatar= "🤷" if speaker.startswith("Você") else "👩‍🌾").markdown(msg)
    # st.success("Ouvindo...")

    with st.spinner("Aguarde..."):
        # salvar temporário
        temp_audio = "temp_audio.wav"
        with open(temp_audio, "wb") as f:
            f.write(audio_bytes)

        with open(temp_audio, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

    user_text = transcription.text
    st.write(f" **{user_text}**")

    with st.spinner("Analisando..."):
        main_content = add_memory_entry(user_text)
        resposta = rag_pipeline(user_text, main_content)

    st.session_state.chat_history.append(("Você (áudio)", user_text))
    st.session_state.chat_history.append(("Assistente", resposta))


# -----------------------------------------------------
# HISTÓRICO DO CHAT
# -----------------------------------------------------
with placeholder.container(height=550):
    if st.session_state.chat_history:
        for speaker, msg in st.session_state.chat_history:
            st.chat_message("user" if speaker.startswith("Você") else "assistant", avatar= "🤷" if speaker.startswith("Você") else "👩‍🌾").markdown(msg)
    else:
        st.image("health-clipart.png", width=650)
    