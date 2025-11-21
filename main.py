import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

st.set_page_config(page_title="Assistente do Restaurante Bella Vita", page_icon="🤖")

st.title("Restaurante Bella Vita — Chat do Cardápio")
st.markdown("Converse com o cardápio e descubra pratos, ingredientes e recomendações!")

if not os.path.exists("instruct.txt"):
    st.warning("Arquivo `instruct.txt` não encontrado! Crie um arquivo com o cardápio.")
    st.stop()

loader = TextLoader("instruct.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    url=qdrant_url,
    collection_name="restaurant_menu"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_template(
    """
    Você é um assistente de restaurante. Responda à pergunta do cliente com base no cardápio abaixo.

    Cardápio:
    {context}

    Pergunta: {question}
    """
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def rag_pipeline(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    chain_input = {"context": context, "question": question}
    response = llm.invoke(prompt.format(**chain_input))
    return response.content

st.divider()
user_input = st.chat_input("Faça uma pergunta sobre o cardápio...")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if user_input:
    with st.spinner("🍽️ Consultando o cardápio..."):
        resposta = rag_pipeline(user_input)
        st.session_state.chat_history.append(("Você", user_input))
        st.session_state.chat_history.append(("Assistente", resposta))

for speaker, msg in st.session_state.chat_history:
    st.chat_message("user" if speaker == "Você" else "assistant").markdown(msg)
