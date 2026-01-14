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

# LangChain + Qdrant - IMPORTS CORRIGIDOS
from typing import List, Optional
from langchain_core.documents import Document  # MUDANÇA AQUI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# Importar o avaliador
from llm_evaluator import LLMEvaluator

# -----------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# -----------------------------------------------------

load_dotenv()
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Inicializar avaliador
evaluator = LLMEvaluator()

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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# prompt = ChatPromptTemplate.from_template(
#     """
# Você é um assistente especializado em bem-estar natural.

# MEMÓRIA DO USUÁRIO (HISTÓRICO DE SINTOMAS):
# {user_memory}

# CONTEXTOS RECUPERADOS DO RAG:
# {context}

# INSTRUÇÕES:
# - Utilize o histórico quando relevante.
# - Utilize o contexto técnico do RAG quando necessário.
# - Comente sobre recorrência de sintomas quando aplicável.
# - Não invente informações.
# - Se não houver dados suficientes, diga isso claramente.

# Pergunta:
# {question}
#     """
# )

prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado em bem-estar natural e práticas integrativas de saúde.

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

INSTRUÇÕES DA RESPOSTA:

1. ANÁLISE DE HISTÓRICO:
   - Identifique padrões e recorrências de sintomas
   - Mencione explicitamente quando houver sintomas repetidos
   - Considere a frequência e duração dos sintomas relatados

2. USO DO CONHECIMENTO TÉCNICO:
   - Base suas recomendações EXCLUSIVAMENTE no contexto fornecido
   - Cite as fontes quando mencionar informações técnicas
   - Use linguagem acessível para explicar conceitos complexos

3. SEGURANÇA E RESPONSABILIDADE:
   - NUNCA substitua orientação médica profissional
   - Recomende buscar um profissional de saúde para sintomas graves, persistentes ou preocupantes
   - Deixe claro quando uma informação está além do seu escopo
   - Não faça diagnósticos ou prescreva tratamentos

4. QUALIDADE DA RESPOSTA:
   - Seja específico e prático nas recomendações
   - Organize a resposta em tópicos quando apropriado
   - Inclua contraindicações e precauções relevantes
   - Se não houver informações suficientes, admita claramente

5. TOM E ESTILO:
   - Seja empático e acolhedor
   - Use linguagem clara e objetiva
   - Evite jargões médicos sem explicação
   - Demonstre cuidado genuíno com o bem-estar do usuário

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

    chain_input = {
        "context": context,
        "question": question,
        "user_memory": memory_text if memory_text else "Sem histórico armazenado."
    }
    
    print("CHAIN INPUT:", chain_input)

    response = llm.invoke(prompt.format(**chain_input))
    
    # AVALIAÇÃO COM RAGAS (sem ground_truth)
    contexts_for_eval = [doc[0].page_content for doc in qdocs]
    metrics = evaluator.evaluate_response(
        question=question,
        llm_answer=response.content,
        contexts=contexts_for_eval
    )
    
    # Salvar métricas
    metrics["question"] = question
    metrics["answer"] = response.content
    evaluator.save_metrics(metrics)
    
    return response.content, metrics


# -----------------------------------------------------
# INTERFACE (TEXTO + ÁUDIO)
# -----------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_metrics" not in st.session_state:
    st.session_state.show_metrics = False

# Create a placeholder
placeholder = st.empty()
 
with st.container():
    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000, text="",
        recording_color="#e8352c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="1x",)

    user_input = st.chat_input("Como posso ajudar hoje?") 

# Sidebar com métricas
with st.sidebar:
    st.header("📊 Métricas de Qualidade")
    
    # Obter últimas métricas do histórico
    last_metrics = None
    if st.session_state.chat_history:
        for speaker, msg, metrics in reversed(st.session_state.chat_history):
            if metrics:
                last_metrics = metrics
                break
    
    if last_metrics:
        st.subheader("Última resposta:")
        
        # Score composto
        composite = last_metrics.get("composite_score", 0)
        quality = last_metrics.get("quality_rating", "N/A")
        st.metric("📊 Score Geral", f"{composite:.2f}", quality)
        
        st.divider()
        
        # Métricas RAGAS
        st.markdown("**🔍 Métricas RAGAS**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Relevância", f"{last_metrics.get('answer_relevancy', 0):.2f}")
        with col2:
            st.metric("✅ Fidelidade", f"{last_metrics.get('faithfulness', 0):.2f}")
        
        st.divider()
        
        # Métricas de Saúde
        st.markdown("**🏥 Métricas de Saúde**")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("🛡️ Segurança", f"{last_metrics.get('safety', 0):.2f}")
            st.metric("📋 Completude", f"{last_metrics.get('completeness', 0):.2f}")
            st.metric("📚 Fundamentação", f"{last_metrics.get('source_attribution', 0):.2f}")
        with col4:
            st.metric("🎯 Precisão", f"{last_metrics.get('medical_accuracy', 0):.2f}")
            st.metric("⚡ Acionabilidade", f"{last_metrics.get('actionability', 0):.2f}")
        
        # Mostrar issues críticas se existirem
        critical_issues = last_metrics.get('critical_issues', [])
        if critical_issues:
            st.warning("⚠️ **Problemas Críticos Detectados:**")
            for issue in critical_issues:
                st.write(f"- {issue}")
        
        st.divider()
        
        # Botão para gerar relatório
        if st.button("📊 Gerar Relatório Completo"):
            report = evaluator.generate_report()
            if report:
                st.json(report)
                
        # DEBUG: Mostrar todas as métricas disponíveis
        with st.expander("🔍 Ver todas as métricas (DEBUG)"):
            st.json(last_metrics)
    else:
        st.info("💬 Faça uma pergunta para ver as métricas de qualidade")

# -----------------------------------------------------
# ENTRADA POR TEXTO
# -----------------------------------------------------

if user_input:
    # Adicionar mensagem do usuário imediatamente
    st.session_state.chat_history.append(("Você", user_input, None))
    
    # Atualizar display
    with placeholder.container(height=550):
        for speaker, msg, metrics in st.session_state.chat_history:
            st.chat_message("user" if speaker.startswith("Você") else "assistant", 
                          avatar="🤷" if speaker.startswith("Você") else "👩‍🌾").markdown(msg)

    with st.spinner("Analisando..."):
        add_memory_entry(user_input)
        resposta, metrics = rag_pipeline(user_input)

    # Adicionar resposta com métricas
    st.session_state.chat_history.append(("Assistente", resposta, metrics))
    
    # Forçar rerun para atualizar sidebar
    st.rerun()

# -----------------------------------------------------
# ENTRADA POR ÁUDIO
# -----------------------------------------------------

if audio_bytes:
    with st.spinner("Transcrevendo áudio..."):
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
    
    # Adicionar mensagem do usuário
    st.session_state.chat_history.append(("Você (áudio)", user_text, None))
    
    # Atualizar display
    with placeholder.container(height=550):
        for speaker, msg, metrics in st.session_state.chat_history:
            st.chat_message("user" if speaker.startswith("Você") else "assistant",
                          avatar="🤷" if speaker.startswith("Você") else "👩‍🌾").markdown(msg)

    with st.spinner("Analisando..."):
        add_memory_entry(user_text)
        resposta, metrics = rag_pipeline(user_text)

    # Adicionar resposta com métricas
    st.session_state.chat_history.append(("Assistente", resposta, metrics))
    
    # Forçar rerun para atualizar sidebar
    st.rerun()


# -----------------------------------------------------
# HISTÓRICO DO CHAT
# -----------------------------------------------------
with placeholder.container(height=550):
    if st.session_state.chat_history:
        for speaker, msg, metrics in st.session_state.chat_history:
            st.chat_message("user" if speaker.startswith("Você") else "assistant", avatar= "🤷" if speaker.startswith("Você") else "👩‍🌾").markdown(msg)
    else:
        st.image("health-clipart.png", width=650)
