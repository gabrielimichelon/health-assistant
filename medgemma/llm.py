import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Configuração da página
st.set_page_config(
    page_title="Assistente Médico Leve",
    page_icon="🏥",
    layout="wide"
)

# Função para limpar memória
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Função para carregar o modelo
@st.cache_resource
def load_model(model_choice):
    # Dicionário de modelos disponíveis
    models = {
        "Gemma 3 1B (Recomendado - 2GB GPU)": {
            "name": "google/gemma-3-1b-it",
            "needs_token": True,
            "description": "Modelo geral leve do Google, ideal para 2GB GPU"
        },
        "Gemma 2 2B": {
            "name": "google/gemma-2-2b-it",
            "needs_token": True,
            "description": "Modelo médio, pode funcionar em 2GB GPU"
        },
        "BioMistral 7B (Médico)": {
            "name": "BioMistral/BioMistral-7B-DARE",
            "needs_token": False,
            "description": "Modelo médico especializado (precisa de mais memória)"
        },
        "Phi-2 2.7B": {
            "name": "microsoft/phi-2",
            "needs_token": False,
            "description": "Modelo compacto da Microsoft, boa performance"
        }
    }
    
    model_info = models[model_choice]
    model_name = model_info["name"]
    
    try:
        st.info(f"📦 Carregando: {model_name}")
        
        # Carrega tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=None,  # Adicione seu token HuggingFace aqui se necessário
            trust_remote_code=True
        )
        
        # Configuração importante para evitar erros CUDA
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Carrega modelo otimizado para GPU pequena
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Usa FP16 para economizar memória
            device_map="auto",
            low_cpu_mem_usage=True,
            token=None,  # Adicione seu token HuggingFace aqui se necessário
            trust_remote_code=True
        )
        
        # Coloca modelo em modo de avaliação
        model.eval()
        
        return model, tokenizer, model_info
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Erro ao carregar modelo: {error_msg}")
        
        if "gated" in error_msg.lower() or "authenticated" in error_msg.lower():
            st.info(f"""
            ⚠️ Este modelo requer autenticação no HuggingFace:
            
            1. Acesse: https://huggingface.co/{model_name}
            2. Clique em "Agree and access repository"
            3. Execute: `huggingface-cli login`
            4. Cole seu token (obtido em: https://huggingface.co/settings/tokens)
            """)
        
        return None, None, None

# Função para gerar resposta
def generate_response(model, tokenizer, prompt, max_length=256, temperature=0.7, model_name=""):
    try:
        # Formata prompt baseado no modelo
        if "gemma" in model_name.lower():
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "phi" in model_name.lower():
            formatted_prompt = f"Instruct: {prompt}\nOutput:"
        else:
            formatted_prompt = f"### Human: {prompt}\n### Assistant:"
        
        # Prepara o input
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True,
            return_attention_mask=True
        )
        
        # Move inputs para o device do modelo
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Gera resposta
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    clear_memory()
                    raise Exception("⚠️ Memória GPU insuficiente! Tente:\n- Reduzir o tamanho da resposta\n- Escolher um modelo menor\n- Limpar a memória GPU")
                raise e
        
        # Decodifica a resposta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Limpa a resposta removendo o prompt
        for marker in [formatted_prompt, prompt, "<start_of_turn>", "<end_of_turn>", 
                       "Instruct:", "Output:", "### Human:", "### Assistant:"]:
            response = response.replace(marker, "")
        
        response = response.strip()
        
        return response if response else "Desculpe, não consegui gerar uma resposta adequada."
    
    except Exception as e:
        clear_memory()
        return f"⚠️ Erro: {str(e)}"

# Interface Streamlit
st.title("🏥 Assistente Médico Otimizado para 2GB GPU")
st.markdown("**⚠️ Aviso**: Este é um assistente educacional. Sempre consulte profissionais de saúde qualificados.")

# Sidebar com configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Seleção de modelo
    model_choice = st.selectbox(
        "Escolha o modelo:",
        [
            "Gemma 3 1B (Recomendado - 2GB GPU)",
            "Gemma 2 2B",
            "BioMistral 7B (Médico)",
            "Phi-2 2.7B"
        ],
        help="Gemma 3 1B é o mais leve e recomendado para GPU de 2GB"
    )
    
    st.markdown("---")
    
    max_length = st.slider(
        "Tamanho da resposta (tokens)",
        min_value=64,
        max_value=512,
        value=256,
        step=64,
        help="⚠️ Reduza se tiver erro de memória"
    )
    
    temperature = st.slider(
        "Temperatura (criatividade)",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    st.markdown("---")
    
    # Prompt de sistema médico
    system_prompt = st.text_area(
        "Prompt do Sistema (Contexto Médico)",
        value="Você é um assistente médico educacional. Forneça informações claras, precisas e baseadas em evidências. Sempre recomende consultar profissionais de saúde para diagnósticos reais.",
        height=150
    )
    
    st.markdown("---")
    
    # Informações sobre memória GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
        
        st.metric("GPU Total", f"{gpu_memory:.1f} GB")
        st.metric("GPU em Uso", f"{gpu_allocated:.1f} GB")
        
        if gpu_allocated > gpu_memory * 0.8:
            st.warning("⚠️ Memória GPU quase cheia!")
    else:
        st.info("💻 Usando CPU (mais lento)")
    
    if st.button("🗑️ Limpar memória GPU"):
        clear_memory()
        st.success("✅ Memória limpa!")

# Área de exemplos
with st.expander("💡 Exemplos de Perguntas", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Sintomas e Condições:**
        - Quais são os sintomas de diabetes tipo 2?
        - O que é hipertensão arterial?
        - Como identificar anemia?
        """)
    
    with col2:
        st.markdown("""
        **Tratamentos e Procedimentos:**
        - Como funciona a insulina?
        - O que é um raio-X de tórax?
        - Diferença entre vírus e bactéria
        """)

# Inicialização do histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_model" not in st.session_state:
    st.session_state.current_model = None

# Verifica se precisa recarregar o modelo
if st.session_state.current_model != model_choice:
    st.session_state.current_model = model_choice
    if "model" in st.session_state:
        del st.session_state["model"]
        del st.session_state["tokenizer"]
        clear_memory()

# Carrega o modelo
if "model" not in st.session_state:
    with st.spinner(f"🔄 Carregando {model_choice}..."):
        model, tokenizer, model_info = load_model(model_choice)
        
        if model is not None and tokenizer is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_info = model_info
            st.success(f"✅ {model_choice} carregado!")
        else:
            st.stop()

# Mostra informações do modelo
st.info(f"**Modelo ativo**: {st.session_state.model_info['description']}")

# Mostra o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta médica..."):
    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostra mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gera e mostra resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("🤔 Pensando..."):
            # Adiciona contexto médico ao prompt
            contextualized_prompt = f"{system_prompt}\n\nPergunta: {prompt}"
            
            # Gera resposta
            response = generate_response(
                st.session_state.model, 
                st.session_state.tokenizer, 
                contextualized_prompt,
                max_length=max_length,
                temperature=temperature,
                model_name=st.session_state.model_info['name']
            )
            
            st.markdown(response)
    
    # Adiciona resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Limpa memória após cada interação
    clear_memory()

# Botões de ação
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Nova conversa"):
        st.session_state.messages = []
        clear_memory()
        st.rerun()

with col2:
    if st.button("💾 Exportar conversa"):
        conversation = "\n\n".join([
            f"**{msg['role'].upper()}**: {msg['content']}" 
            for msg in st.session_state.messages
        ])
        st.download_button(
            label="📥 Baixar TXT",
            data=conversation,
            file_name="conversa_medica.txt",
            mime="text/plain"
        )

with col3:
    if st.button("🔄 Trocar modelo"):
        if "model" in st.session_state:
            del st.session_state["model"]
            del st.session_state["tokenizer"]
            clear_memory()
            st.rerun()

# Informações no rodapé
st.markdown("---")
st.caption(f"🤖 Modelo: {st.session_state.model_info['name']} | Otimizado para GPU de 2GB com FP16")
st.caption("⚠️ Para usar MedGemma 4B, você precisa de ~4-6GB de VRAM. Este assistente usa modelos menores compatíveis com 2GB.")