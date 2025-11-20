import streamlit as st
import os
import sys
import torch
import time
import pandas as pd
import numpy as np
import faiss
import requests
import gc
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration

# --- 0. CRITICAL: ENVIRONMENT SETUP ---
# Pastikan HF Cache mengarah ke D: agar tidak download ulang
os.environ['HF_HOME'] = "/mnt/d/huggingface_cache" 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MEMORY MANAGEMENT ---
def clear_memory():
    """Membersihkan RAM dan VRAM secara paksa."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        h1 {font-family: 'Helvetica Neue', sans-serif; font-weight: 700; letter-spacing: -1px;}
        .stAlert {border-radius: 4px;}
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        .img-caption {font-size: 0.8rem; color: #555;}
        .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 1. BACKEND CONFIGURATION ---
class Config:
    # Path Management: Naik satu level dari folder 'User_Interface'
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    IMAGES_DIR = os.path.join(BASE_DIR, "Dataset", "Images")
    CAPTIONS_FILE = os.path.join(BASE_DIR, "Dataset", "captions.txt")
    
    # [FIXED PATHS] Sesuai permintaan user
    INDEX_PATH = os.path.join(BASE_DIR, "Notebook", "flickr30k_large.index") 
    METADATA_PATH = os.path.join(BASE_DIR, "Notebook", "metadata_large.json")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default Model (Akan di-override otomatis oleh Smart Loader)
    RETRIEVAL_MODEL = "openai/clip-vit-large-patch14"
    CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
    
    # [SMART CONFIG] Coba localhost dulu (Native WSL), kalau gagal baru IP Windows
    LLM_API_LOCAL = "http://localhost:11434/api/generate"
    LLM_API_WIN = "http://172.29.240.1:11434/api/generate"
    
    # Default pointer
    LLM_API = LLM_API_LOCAL 
    LLM_MODEL = "llama3" 

# --- 2. CORE CLASSES (SMART LOADER) ---

@st.cache_resource(show_spinner=False)
def load_retrieval_system():
    """Memuat CLIP Model dan FAISS Index dengan Auto-Detect Dimension."""
    clear_memory()
    
    # A. Cek File Index
    print(f"📂 Checking Index at: {Config.INDEX_PATH}")
    if not os.path.exists(Config.INDEX_PATH):
        st.error(f"⚠️ Index file missing at `{Config.INDEX_PATH}`.\nPlease run the Notebook first to generate the index.")
        return None, None, None
        
    # B. Load Index & Deteksi Dimensi
    try:
        index = faiss.read_index(Config.INDEX_PATH)
        disk_dim = index.d
        print(f"📏 Detected Index Dimension: {disk_dim}")
        
        # Auto-Switch Model berdasarkan dimensi index
        if disk_dim == 512:
            print("Switching to CLIP-BASE (512) to match index.")
            Config.RETRIEVAL_MODEL = "openai/clip-vit-base-patch32"
        elif disk_dim == 768:
            print("Switching to CLIP-LARGE (768) to match index.")
            Config.RETRIEVAL_MODEL = "openai/clip-vit-large-patch14"
        else:
            st.warning(f"Unknown index dimension: {disk_dim}. Keeping default.")
            
    except Exception as e:
        st.error(f"Error reading index: {e}")
        return None, None, None
    
    # C. Load Metadata
    import json
    if not os.path.exists(Config.METADATA_PATH):
         st.error(f"Metadata json missing at {Config.METADATA_PATH}")
         return None, None, None
         
    with open(Config.METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    # D. Load Encoder (Sesuai Config yang sudah disesuaikan)
    print(f"Allocating Encoder: {Config.RETRIEVAL_MODEL}...")
    processor = CLIPProcessor.from_pretrained(Config.RETRIEVAL_MODEL)
    model = CLIPModel.from_pretrained(Config.RETRIEVAL_MODEL).to(Config.DEVICE)
    model.eval()
        
    return processor, model, (index, metadata)

@st.cache_resource(show_spinner=False)
def load_generative_system():
    """Memuat BLIP-2 Model."""
    clear_memory()
    print(f"Loading Gen Model: {Config.CAPTION_MODEL}...")
    
    # Optimasi Loading Float16
    dtype = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    
    processor = Blip2Processor.from_pretrained(Config.CAPTION_MODEL)
    model = Blip2ForConditionalGeneration.from_pretrained(
        Config.CAPTION_MODEL, 
        torch_dtype=dtype
    ).to(Config.DEVICE)
    
    return processor, model

# --- 3. HELPER FUNCTIONS ---

def perform_retrieval(query_text, clip_processor, clip_model, vector_db, k=5):
    index, metadata = vector_db
    
    # Embed Text
    text_with_prompt = [f"A photo of {query_text}"]
    inputs = clip_processor(text=text_with_prompt, return_tensors="pt", padding=True).to(Config.DEVICE)
    
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    
    # Search FAISS
    q_vec = features.cpu().numpy().astype('float32')
    distances, indices = index.search(q_vec, k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            results.append({
                "filename": metadata[idx],
                "score": float(dist),
                "path": os.path.join(Config.IMAGES_DIR, metadata[idx])
            })
            
    # Cleanup tensors
    del inputs, features, q_vec
    clear_memory()
    return results

def generate_context(images_data, blip_processor, blip_model):
    contexts = []
    dtype = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    
    for item in images_data:
        try:
            # Convert absolute path if necessary
            img_path = item['path']
            if not os.path.isabs(img_path):
                img_path = os.path.abspath(img_path)
                
            raw_image = Image.open(img_path).convert('RGB')
            inputs = blip_processor(images=raw_image, return_tensors="pt").to(Config.DEVICE, dtype)
            
            # Beam Search
            with torch.no_grad():
                out = blip_model.generate(
                    **inputs, 
                    max_new_tokens=70,
                    min_length=15,
                    num_beams=5,            # Beam search aktif
                    repetition_penalty=1.5
                )
            
            caption = blip_processor.decode(out[0], skip_special_tokens=True).strip()
            contexts.append(caption)
            
            del inputs, out, raw_image
            
        except Exception as e:
            contexts.append(f"[Error analyzing image: {str(e)}]")
            
    clear_memory()
    return contexts

def check_ollama_status():
    """Cek apakah Ollama jalan di localhost atau IP Windows."""
    urls = [Config.LLM_API_LOCAL, Config.LLM_API_WIN]
    
    for url in urls:
        try:
            # Cek ping tags
            test_url = url.replace("/generate", "/tags")
            resp = requests.get(test_url, timeout=1)
            if resp.status_code == 200:
                return True, url # Found working URL
        except:
            continue
    return False, None

def query_llm(context_list, user_query, api_url):
    context_str = "\n".join([f"- Image {i+1}: {txt}" for i, txt in enumerate(context_list)])
    
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an intelligent visual assistant. Answer the question based ONLY on the image descriptions below.
    
    CONTEXT:
    {context_str}
    
    QUESTION: "{user_query}"
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    payload = {"model": Config.LLM_MODEL, "prompt": prompt, "stream": False}
    
    # Fungsi internal untuk mengirim request
    def send_request():
        return requests.post(api_url, json=payload, timeout=120)

    try:
        response = send_request()
        
        # [FIX] Jika Error 500 (Ollama Crash), coba sekali lagi (Auto-Retry)
        if response.status_code == 500:
            time.sleep(2) # Beri waktu napas
            response = send_request() # Coba lagi
            
        if response.status_code == 200:
            return response.json().get("response", "Error: Empty response from Ollama.")
        elif response.status_code == 404:
            # Fallback ke latest
            payload['model'] = "llama3:latest"
            retry = requests.post(api_url, json=payload, timeout=120)
            if retry.status_code == 200: return retry.json().get("response", "")
            return f"Error 404: Model llama3 not found on server."
        else:
            return f"Error: LLM Service returned status {response.status_code}"
            
    except Exception as e:
        return f"Connection Error: {e}"

# --- 4. MAIN UI LAYOUT ---

def main():
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Retrieval Count (K)", min_value=1, max_value=10, value=3)
        
        st.markdown("---")
        
        # [TAMBAHAN 1] Penjelasan Vector Search
        with st.expander("📐 Vector Search Method", expanded=True):
            st.markdown("""
            **Metric:** Cosine Similarity
            
            **Implementation:**
            $$
            \\text{Sim}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}
            $$
            
            *Notes:*
            Vectors are L2-normalized (`||A||=1`), so the Dot Product in FAISS (`IndexFlatIP`) equals Cosine Similarity.
            """)
        
        st.markdown("---")
        if st.button("🧹 Flush RAM/VRAM"):
            clear_memory()
            st.toast("Memory cleared!", icon="🧹")
            
        st.divider()
        
        with st.status("System Status", expanded=True) as status:
            # 1. Load Backend
            clip_proc, clip_model, vector_db = load_retrieval_system()
            if vector_db:
                st.success(f"✅ Retrieval Ready ({Config.RETRIEVAL_MODEL})")
            else:
                st.error("❌ Retrieval Engine Failed")
                st.stop()

            # 2. Load Generative
            blip_proc, blip_model = load_generative_system()
            st.success("✅ Generative Engine Ready")
            
            # 3. Check Ollama
            ollama_ok, working_url = check_ollama_status()
            if ollama_ok:
                st.success(f"✅ Llama-3 Connected")
                Config.LLM_API = working_url # Update Config
            else:
                st.warning("⚠️ Ollama Offline")
            
            status.update(label="System Operational", state="complete", expanded=False)
        
        st.info(
            "**About:**\n"
            "This system utilizes RAG (Retrieval Augmented Generation) to answer questions based on visual evidence from the Flickr30k dataset."
        )

    # Main Content
    st.title("Multimodal RAG System")
    st.markdown("##### A RAG-Based Approach to Image Retrieval and Context-Aware Generation")
    
    # Input Area
    query = st.text_input("Enter your visual query:", placeholder="e.g., Two men playing guitar in the park...")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_btn and query:
        if not query.strip():
            st.warning("Please enter a valid query.")
            return

        # --- PIPELINE EXECUTION ---
        clear_memory()
        start_time = time.time()
        
        # 1. Retrieval Phase
        with st.spinner("🔍 Searching Vector Database..."):
            ret_start = time.time()
            results = perform_retrieval(query, clip_proc, clip_model, vector_db, k=top_k)
            retrieval_latency = time.time() - ret_start

        # Hitung Metrik
        scores = [r['score'] for r in results]
        avg_score = np.mean(scores) if scores else 0
        top_score = scores[0] if scores else 0
        
        # [TAMBAHAN 2] Matrix Pengukuran Performa
        st.markdown("### 📊 Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval Latency", f"{retrieval_latency*1000:.1f} ms")
        m2.metric("Top-1 Similarity", f"{top_score:.4f}")
        m3.metric("Avg Confidence", f"{avg_score:.4f}")
        m4.metric("Retrieval Count", f"{len(results)} items")
        
        st.divider()

        # Layout: Columns for Results
        col_left, col_right = st.columns([1.2, 1])

        # 2. Display Retrieval Results (Left Column)
        with col_left:
            st.subheader(f"Evidence (Top-{top_k})")
            
            # Display as a clean grid or list
            tabs = st.tabs([f"Rank {i+1}" for i in range(len(results))])
            
            for i, (tab, item) in enumerate(zip(tabs, results)):
                with tab:
                    try:
                        img = Image.open(item['path'])
                        # [FIX] Ganti deprecated use_column_width -> use_container_width
                        st.image(img, use_container_width=True)
                        
                        # Metadata Container
                        with st.expander("See Metadata", expanded=True):
                            st.code(f"Filename: {item['filename']}", language="text")
                            st.progress(item['score'], text=f"Similarity Score: {item['score']:.4f}")
                    except Exception as e:
                        st.error(f"Image load error: {e}")

        # 3. Generative Phase (Right Column)
        with col_right:
            st.subheader("Context-Aware Generation")
            
            # BLIP-2 Processing
            with st.status("Analyzing Visual Context...", expanded=True) as gen_status:
                st.write("👁️ Generating captions with BLIP-2...")
                visual_contexts = generate_context(results, blip_proc, blip_model)
                
                st.write("🧠 Reasoning with Llama-3...")
                if ollama_ok:
                    gen_start = time.time()
                    final_answer = query_llm(visual_contexts, query, Config.LLM_API)
                    gen_latency = time.time() - gen_start
                else:
                    final_answer = "⚠️ Ollama server is not reachable. Please start 'ollama serve' in terminal."
                    gen_latency = 0
                
                gen_status.update(label="Generation Complete", state="complete", expanded=False)
            
            # Display Answer
            st.markdown("### AI Response")
            if "Error" in final_answer:
                st.error(final_answer)
            else:
                st.success(final_answer)
                st.caption(f"Generation Latency: {gen_latency:.2f}s")
            
            # Display Generated Contexts (Transparency)
            with st.expander("View Generated Visual Contexts (BLIP-2 Output)"):
                for idx, ctx in enumerate(visual_contexts):
                    st.text(f"Img {idx+1}: {ctx}")

        # Total Time Footer
        total_time = time.time() - start_time
        st.divider()
        st.caption(f"Total Pipeline Latency: {total_time:.2f} seconds | Powered by CLIP, FAISS, BLIP-2, & Llama-3")
        
        clear_memory()

if __name__ == "__main__":
    main()