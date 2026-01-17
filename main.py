import os
import json
import tempfile
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Áudio
from audio_recorder_streamlit import audio_recorder

# Google Gemini
import google.generativeai as genai

# LangChain + Qdrant
from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore


# -----------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# -----------------------------------------------------

load_dotenv()
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_TCC")

print("Google API Key:", GOOGLE_API_KEY is not None)

# Configurar Gemini
genai.configure(api_key=GOOGLE_API_KEY)

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

def add_memory_entry(user_message):
    """Extrai sintomas e salva com a data."""
    model = genai.GenerativeModel('gemini-3-flash-preview')

    extract_prompt = f"""
    Extraia sintomas mencionados no texto abaixo e retorne como lista JSON.

    Texto: "{user_message}"

    Responda apenas no formato:
    ["sintoma1", "sintoma2", ...]
    """

    try:
        response = model.generate_content(extract_prompt)
        extracted = response.text.strip()
        # Remove markdown code blocks se existir
        if extracted.startswith("```"):
            extracted = extracted.split("```")[1]
            if extracted.startswith("json"):
                extracted = extracted[4:]
        symptoms = json.loads(extracted.strip())
    except Exception as e:
        print(f"Erro ao extrair sintomas: {e}")
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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Usar ChatGoogleGenerativeAI do LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

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

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embedding_model,
    )
    
    results= vector_store.similarity_search_with_relevance_scores(query=query, k=k)
    return results


def rag_pipeline(question):
    memory = load_memory()

    memory_text = "\n".join(
        [f"- {m['date']}: {m['text']} (sintomas: {', '.join(m['symptoms'])})"
         for m in memory[-2:]]
    )

    qdocs = search_similar_documents(query=question, embedding_model=embeddings)
    context = "\n\n".join([doc[0].page_content for doc in qdocs])
    
    print("Context retrieved from RAG:", context)

    chain_input = {
        "context": context,
        "question": question,
        "user_memory": memory_text if memory_text else "Sem histórico armazenado."
    }

    # Criar a chain completa
    chain = prompt | llm
    
    # Invocar e retornar apenas o conteúdo
    response = chain.invoke(chain_input)
    
    # Garantir que retorna apenas o texto da resposta
    if hasattr(response, 'text'):
        return response.text
    elif isinstance(response, str):
        return response
    else:
        return str(response)


# -----------------------------------------------------
# TRANSCRIÇÃO DE ÁUDIO COM GEMINI
# -----------------------------------------------------

def transcribe_audio_gemini(audio_path: str) -> str:
    """Transcreve áudio usando Gemini."""
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        # Upload do arquivo de áudio
        audio_file = genai.upload_file(audio_path)
        
        # Solicitar transcrição
        response = model.generate_content([
            "Transcreva o áudio em português. Retorne apenas o texto transcrito, sem comentários adicionais.",
            audio_file
        ])
        
        return response.text.strip()
    except Exception as e:
        print(f"Erro na transcrição: {e}")
        return ""


# -----------------------------------------------------
# INTERFACE (TEXTO + ÁUDIO)
# -----------------------------------------------------

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
        # PRIMEIRO: gerar a resposta
        resposta = rag_pipeline(user_input)
        
        # DEPOIS: salvar na memória
        add_memory_entry(user_input)

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

    with st.spinner("Aguarde..."):
        # salvar temporário
        temp_audio = "temp_audio.wav"
        with open(temp_audio, "wb") as f:
            f.write(audio_bytes)

        # Transcrever com Gemini
        user_text = transcribe_audio_gemini(temp_audio)

    if user_text:
        st.write(f" **{user_text}**")

        with st.spinner("Analisando..."):
            # PRIMEIRO: gerar a resposta
            resposta = rag_pipeline(user_text)
            
            # DEPOIS: salvar na memória
            add_memory_entry(user_text)

        st.session_state.chat_history.append(("Você (áudio)", user_text))
        st.session_state.chat_history.append(("Assistente", resposta))
    else:
        st.error("Não foi possível transcrever o áudio.")


# -----------------------------------------------------
# HISTÓRICO DO CHAT
# -----------------------------------------------------
with placeholder.container(height=550):
    if st.session_state.chat_history:
        for speaker, msg in st.session_state.chat_history:
            st.chat_message("user" if speaker.startswith("Você") else "assistant", avatar= "🤷" if speaker.startswith("Você") else "👩‍🌾").markdown(msg)
    else:
        st.image("health-clipart.png", width=650)
