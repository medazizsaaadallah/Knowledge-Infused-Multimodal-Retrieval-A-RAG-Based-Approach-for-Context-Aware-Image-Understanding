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
import json
from PIL import Image
from collections import defaultdict
from difflib import SequenceMatcher
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration

# [CRITICAL] Library untuk PEFT (Fine-Tuning) & Metrik Akademis
from peft import PeftModel

# Import NLTK untuk BLEU Score
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    import os
    os.system('pip install nltk')
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Import ROUGE untuk Metrik Tambahan (Opsional, fallback jika tidak ada)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

# --- 0. ENVIRONMENT SETUP ---
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
    """
    Forces garbage collection and clears CUDA cache to manage memory usage efficiently.
    Crucial for preventing OOM errors on GPUs with limited VRAM.
    """
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
        .gt-caption {font-size: 0.85rem; color: #2e7d32; border-left: 2px solid #2e7d32; padding-left: 10px; margin-bottom: 5px;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 1. BACKEND CONFIGURATION ---
class Config:
    """
    Centralized configuration for file paths, model identifiers, and API endpoints.
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    IMAGES_DIR = os.path.join(BASE_DIR, "Dataset", "Images")
    CAPTIONS_FILE = os.path.join(BASE_DIR, "Dataset", "captions.txt")
    INDEX_PATH = os.path.join(BASE_DIR, "Notebook", "flickr30k_large.index") 
    METADATA_PATH = os.path.join(BASE_DIR, "Notebook", "metadata_large.json")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RETRIEVAL_MODEL = "openai/clip-vit-large-patch14"
    CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
    LLM_API_LOCAL = "http://localhost:11434/api/generate"
    LLM_API_WIN = "http://172.29.240.1:11434/api/generate"
    LLM_API = LLM_API_LOCAL 
    LLM_MODEL = "llama3" 

# --- 2. CORE CLASSES (SMART LOADER) ---

@st.cache_resource(show_spinner=False)
def load_captions_dataset():
    """
    Loads Ground Truth captions from CSV/TXT for validation purposes.
    Returns: defaultdict mapping image filenames to list of captions.
    """
    print(f"📂 Loading Captions from: {Config.CAPTIONS_FILE}")
    captions_dict = defaultdict(list)
    if not os.path.exists(Config.CAPTIONS_FILE):
        st.warning(f"⚠️ Captions file not found at {Config.CAPTIONS_FILE}")
        return captions_dict
    try:
        with open(Config.CAPTIONS_FILE, 'r', encoding='utf-8') as f:
            next(f) 
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    img_name, cap = parts
                    clean_cap = cap.strip().strip('"')
                    captions_dict[img_name].append(clean_cap)
        return captions_dict
    except Exception as e:
        st.error(f"Error reading captions file: {e}")
        return captions_dict

@st.cache_resource(show_spinner=False)
def load_retrieval_system():
    """
    Initializes CLIP Model and FAISS Index for semantic search.
    Handles dimension mismatch handling (512 vs 768 dim).
    """
    clear_memory()
    if not os.path.exists(Config.INDEX_PATH):
        st.error(f"⚠️ Index file missing at `{Config.INDEX_PATH}`.")
        return None, None, None
    try:
        index = faiss.read_index(Config.INDEX_PATH)
        disk_dim = index.d
        if disk_dim == 512:
            Config.RETRIEVAL_MODEL = "openai/clip-vit-base-patch32"
        elif disk_dim == 768:
            Config.RETRIEVAL_MODEL = "openai/clip-vit-large-patch14"
    except Exception as e:
        st.error(f"Error reading index: {e}")
        return None, None, None
    if not os.path.exists(Config.METADATA_PATH):
         st.error(f"Metadata json missing at {Config.METADATA_PATH}")
         return None, None, None
    with open(Config.METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    print(f"Allocating Encoder: {Config.RETRIEVAL_MODEL}...")
    processor = CLIPProcessor.from_pretrained(Config.RETRIEVAL_MODEL)
    model = CLIPModel.from_pretrained(Config.RETRIEVAL_MODEL).to(Config.DEVICE)
    model.eval()
    return processor, model, (index, metadata)

@st.cache_resource(show_spinner=False)
def load_generative_system():
    """
    Initializes BLIP-2 Model for Image Captioning.
    Automatically detects and loads Fine-Tuned LoRA Adapter if available.
    """
    clear_memory()
    dtype = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    
    print(f"Loading Base Gen Model: {Config.CAPTION_MODEL}...")
    
    # 1. Load Base Model
    processor = Blip2Processor.from_pretrained(Config.CAPTION_MODEL)
    model = Blip2ForConditionalGeneration.from_pretrained(
        Config.CAPTION_MODEL, torch_dtype=dtype
    ).to(Config.DEVICE)
    
    # 2. Check and Load Fine-Tuned Adapter (LoRA)
    ADAPTER_PATH = os.path.join(Config.BASE_DIR, "fine_tuned_blip2_adapter")
    
    if os.path.exists(ADAPTER_PATH):
        print(f"✅ FOUND FINE-TUNED ADAPTER! Loading from: {ADAPTER_PATH}")
        try:
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            print("🚀 Successfully loaded Fine-Tuned BLIP-2.")
        except Exception as e:
            st.warning(f"⚠️ Failed to load adapter: {e}. Using base model.")
    else:
        print("ℹ️ No adapter found. Using standard base model.")
        
    return processor, model

# --- 3. HELPER FUNCTIONS ---

def perform_retrieval(query_text, clip_processor, clip_model, vector_db, k=5):
    """
    Executes dense vector retrieval using CLIP embeddings and FAISS.
    """
    index, metadata = vector_db
    text_with_prompt = [f"A photo of {query_text}"]
    inputs = clip_processor(text=text_with_prompt, return_tensors="pt", padding=True).to(Config.DEVICE)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
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
    del inputs, features, q_vec
    clear_memory()
    return results

def generate_context(images_data, blip_processor, blip_model):
    """
    Generates visual descriptions using BLIP-2.
    Uses safe generation parameters to avoid OOM on limited hardware.
    """
    contexts = []
    dtype = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    for item in images_data:
        try:
            img_path = item['path']
            if not os.path.isabs(img_path): img_path = os.path.abspath(img_path)
            raw_image = Image.open(img_path).convert('RGB')
            inputs = blip_processor(images=raw_image, return_tensors="pt").to(Config.DEVICE, dtype)
            
            with torch.no_grad():
                out = blip_model.generate(
                    **inputs, max_new_tokens=100, min_length=15,
                    do_sample=True, top_p=0.90, temperature=0.6,
                    repetition_penalty=1.2, num_return_sequences=5 
                )
            
            all_candidates = blip_processor.batch_decode(out, skip_special_tokens=True)
            unique_captions = []
            for cap in all_candidates:
                clean_cap = cap.strip()
                if not clean_cap: continue
                is_duplicate = False
                for existing in unique_captions:
                    similarity = SequenceMatcher(None, clean_cap.lower(), existing.lower()).ratio()
                    if similarity > 0.8: 
                        is_duplicate = True
                        break
                if not is_duplicate: unique_captions.append(clean_cap)
            
            if not unique_captions: unique_captions = ["Image content unclear."]
            main_paragraph = unique_captions[0]
            remaining_points = unique_captions[1:] 
            if remaining_points:
                bullet_points = "\n".join([f"• {cap}" for cap in remaining_points])
                final_text = f"{main_paragraph}\n\n{bullet_points}"
            else:
                final_text = main_paragraph
            contexts.append(final_text)
            del inputs, out, raw_image
            clear_memory()
        except Exception as e:
            contexts.append(f"[Error: {str(e)}]")
            clear_memory()
    return contexts

def check_ollama_status():
    """Checks availability of the Local LLM API."""
    urls = [Config.LLM_API_LOCAL, Config.LLM_API_WIN]
    for url in urls:
        try:
            test_url = url.replace("/generate", "/tags")
            resp = requests.get(test_url, timeout=1)
            if resp.status_code == 200: return True, url
        except: continue
    return False, None

def query_llm(context_list, user_query, api_url):
    """
    Orchestrates the Reasoning Phase using Llama-3.
    Constructs a structured prompt for data auditing.
    """
    full_report = ""
    progress_bar = st.progress(0, text="🕵️ Llama-3 is auditing evidence...")
    
    for i, context_item in enumerate(context_list):
        rank = i + 1
        progress_bar.progress((i + 1) / len(context_list), text=f"Auditing Rank {rank}/{len(context_list)}...")
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Strict Data Auditor. Verify if ONE specific image matches the User Query based on provided evidence.
GOLDEN RULES:
1. TRUST DATASET FACTS (Ground Truth).
2. IF FACTS CONFIRM QUERY DETAILS -> MATCH.
3. IF FACTS CONTRADICT -> NO MATCH.
OUTPUT FORMAT:
**RANK {rank} Analysis:**
- **Verdict:** [MATCH / PARTIAL / NO MATCH]
- **Audit:**
  - Query asked for: "[Keyword]"
  - Dataset Facts say: "[Quote Fact]"
  - AI Vision says: "[Quote Vision]"
- **Conclusion:** [Reasoning]
CONTEXT DATA:
{context_item}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: "{user_query}"
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        payload = {
            "model": Config.LLM_MODEL, 
            "prompt": prompt, 
            "stream": False,
            "options": {"num_predict": 512, "temperature": 0.0, "top_p": 1.0}
        }
        
        def send_request():
            return requests.post(api_url, json=payload, timeout=120)

        try:
            response = send_request()
            if response.status_code == 500:
                time.sleep(1); response = send_request()
            
            if response.status_code == 200:
                full_report += response.json().get("response", "").strip() + "\n\n" + ("-"*40) + "\n\n"
            elif response.status_code == 404:
                full_report += f"**RANK {rank}:** Error - Model not found.\n\n"
            else:
                full_report += f"**RANK {rank}:** Error - API Failure.\n\n"
        except Exception as e:
            full_report += f"**RANK {rank}:** Connection Error: {e}\n\n"
    
    progress_bar.empty()
    return full_report

def calculate_rag_metrics(user_query, generated_answer, retrieved_contexts, clip_processor, clip_model):
    """
    Calculates RAG Quality Metrics using CLIP Latent Space embeddings.
    
    Metrics:
    1. Answer Relevance: Cosine Similarity between (User Query) and (LLM Answer).
    2. Faithfulness: Cosine Similarity between (Visual Context) and (LLM Answer).
    
    Returns: (relevance_score, faithfulness_score)
    """
    try:
        # Clean markdown artifacts from answer
        clean_answer = generated_answer.replace("*", "").replace("#", "").strip()
        
        # Aggregate visual contexts into one string
        context_text = " ".join([ctx.split('\n')[0] for ctx in retrieved_contexts]) 
        
        # Prepare inputs (Truncate to 77 tokens to fit CLIP context window)
        texts = [
            user_query,    # Index 0
            clean_answer,  # Index 1
            context_text   # Index 2
        ]
        
        inputs = clip_processor(
            text=texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(Config.DEVICE)
        
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            
            # 1. Answer Relevance (Query vs Answer)
            relevance = torch.nn.functional.cosine_similarity(feats[0], feats[1], dim=0).item()
            
            # 2. Faithfulness (Context vs Answer)
            faithfulness = torch.nn.functional.cosine_similarity(feats[2], feats[1], dim=0).item()
            
        return relevance, faithfulness
    except Exception as e:
        print(f"RAG Metric Error: {e}")
        return 0.0, 0.0

def calculate_caption_metrics(reference_texts, candidate_text):
    """
    Calculates Generation Metrics: BLEU-4 and ROUGE-L.
    
    Args:
        reference_texts (list of list of str): Tokenized ground truth captions.
        candidate_text (list of str): Tokenized generated caption.
        
    Returns:
        dict: Containing 'bleu4' and 'rougeL' scores.
    """
    metrics = {"bleu4": 0.0, "rougeL": 0.0}
    
    # 1. BLEU-4
    try:
        if reference_texts and candidate_text:
            chencherry = SmoothingFunction()
            metrics["bleu4"] = sentence_bleu(reference_texts, candidate_text, smoothing_function=chencherry.method1)
    except: pass
    
    # 2. ROUGE-L (Using simple implementation if library missing or wrapper)
    try:
        if ROUGE_AVAILABLE:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            # Join tokens back to string for ROUGE
            ref_str = " ".join(reference_texts[0]) 
            cand_str = " ".join(candidate_text)
            scores = scorer.score(ref_str, cand_str)
            metrics["rougeL"] = scores['rougeL'].fmeasure
    except: pass
    
    return metrics

# --- 4. MAIN UI LAYOUT ---

def main():
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Retrieval Count (K)", min_value=1, max_value=10, value=3)
        st.markdown("---")
        with st.expander("📐 Vector Search Method", expanded=True):
            st.markdown("""
            **Metric:** Cosine Similarity
            **Formula:**
            $$
            \\text{Sim}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}
            $$
            """)
        st.markdown("---")
        if st.button("🧹 Flush RAM/VRAM"):
            clear_memory(); st.toast("Memory cleared!", icon="🧹")
        st.divider()
        with st.status("System Status", expanded=True) as status:
            gt_captions = load_captions_dataset()
            st.success(f"✅ Captions Loaded ({len(gt_captions)} items)")
            clip_proc, clip_model, vector_db = load_retrieval_system()
            if vector_db: st.success(f"✅ Retrieval Ready")
            else: st.stop()
            blip_proc, blip_model = load_generative_system()
            st.success("✅ Generative Ready")
            ollama_ok, working_url = check_ollama_status()
            if ollama_ok:
                st.success(f"✅ Llama-3 Connected")
                Config.LLM_API = working_url 
            else: st.warning("⚠️ Ollama Offline")
            status.update(label="System Operational", state="complete", expanded=False)
        st.info("**About:** Multimodal RAG System using Flickr30k.")

    st.title("Multimodal RAG System")
    st.markdown("##### A RAG-Based Approach to Image Retrieval and Context-Aware Generation")
    
    query = st.text_input("Enter your visual query:", placeholder="e.g., Two men playing guitar...")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_btn and query:
        if not query.strip(): st.warning("Please enter a valid query."); return
        clear_memory()
        start_time = time.time()
        
        # 1. RETRIEVAL PHASE
        with st.spinner("🔍 Searching Vector Database..."):
            results = perform_retrieval(query, clip_proc, clip_model, vector_db, k=top_k)

        # METRICS: GT Match Rate Calculation
        scores = [r['score'] for r in results]
        relevant_count = 0
        query_words = set(query.lower().split())
        
        for item in results:
            fname = item['filename']
            is_relevant = False
            if fname in gt_captions:
                for cap in gt_captions[fname]:
                    cap_words = set(cap.lower().split())
                    overlap = len(query_words.intersection(cap_words))
                    if len(query_words) > 0 and (overlap / len(query_words) >= 0.5):
                        is_relevant = True
                        break
            if is_relevant: relevant_count += 1
        
        gt_match_rate = relevant_count / len(results) if results else 0

        # DISPLAY RETRIEVAL METRICS
        st.markdown("### 📊 Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval Latency", f"{(time.time()-start_time)*1000:.1f} ms")
        if scores: m2.metric(f"Sim Score Range", f"{scores[0]:.3f} - {scores[-1]:.3f}")
        else: m2.metric("Sim Score", "0.000")
        m3.metric(f"Avg Sim (Top-{top_k})", f"{np.mean(scores):.4f}")
        m4.metric(f"GT Match (Top-{top_k})", f"{gt_match_rate:.0%}", help="Precision based on Ground Truth matching")
        st.divider()

        col_left, col_right = st.columns([1.2, 1])

        # 2. EVIDENCE DISPLAY
        with col_left:
            st.subheader(f"Evidence (Top-{top_k})")
            tabs = st.tabs([f"Rank {i+1}" for i in range(len(results))])
            for i, (tab, item) in enumerate(zip(tabs, results)):
                with tab:
                    try:
                        img = Image.open(item['path'])
                        st.image(img, use_container_width=True)
                        fname = item['filename']
                        st.markdown("**📂 Ground Truth Captions:**")
                        if fname in gt_captions:
                            with st.container(height=150, border=True):
                                for cap in gt_captions[fname]:
                                    st.markdown(f"<div class='gt-caption'>• {cap}</div>", unsafe_allow_html=True)
                        else: st.caption("No ground truth found.")
                        with st.expander("See Metadata"):
                            st.code(f"File: {fname}\nScore: {item['score']:.4f}")
                    except Exception as e: st.error(f"Image error: {e}")

        # 3. GENERATION & REASONING PHASE
        with col_right:
            st.subheader("Context-Aware Generation")
            with st.status("Processing...", expanded=True) as gen_status:
                st.write("👁️ BLIP-2: Analyzing Visuals...")
                visual_contexts = generate_context(results, blip_proc, blip_model)
                
                st.write("📂 Data Fusion: Merging Vision + Dataset Facts...")
                rich_context_for_llm = []
                for i, blip_text in enumerate(visual_contexts):
                    fname = results[i]['filename']
                    gt_text = "(No verified data available)"
                    if fname in gt_captions:
                        gt_list = gt_captions[fname][:10]
                        gt_text = "\n".join([f"- {c}" for c in gt_list])
                    combined_entry = (
                        f"=== IMAGE #{i+1} (Rank {i+1}) ===\n"
                        f"[AI Vision / BLIP-2 Output]:\n{blip_text}\n\n"
                        f"[Dataset Facts / Ground Truth]:\n{gt_text}\n"
                        f"====================================="
                    )
                    rich_context_for_llm.append(combined_entry)

                st.write("🧠 Llama-3: Auditing Evidence...")
                if ollama_ok:
                    gen_start = time.time()
                    final_answer = query_llm(rich_context_for_llm, query, Config.LLM_API)
                    gen_latency = time.time() - gen_start
                    
                    # [METRIC CALCULATION] RAG Metrics
                    rag_relevance, rag_faithfulness = calculate_rag_metrics(
                        query, final_answer, visual_contexts, clip_proc, clip_model
                    )
                else: 
                    final_answer = "⚠️ Ollama error."; gen_latency = 0
                    rag_relevance, rag_faithfulness = 0.0, 0.0
                
                gen_status.update(label="Done", state="complete", expanded=False)
            
            # Display Final Answer & RAG Metrics
            with st.expander("🤖 View AI Response (Llama-3 Audit)", expanded=True):
                if "Error" in final_answer: st.error(final_answer)
                else: st.markdown(final_answer)
                st.divider()
                
                # Display RAG Metrics
                rm1, rm2, rm3 = st.columns(3)
                rm1.caption(f"Latency: {gen_latency:.2f}s")
                rm2.metric("Answer Relevance", f"{rag_relevance:.4f}", help="Semantic Similarity (Query vs Answer)")
                rm3.metric("Faithfulness", f"{rag_faithfulness:.4f}", help="Semantic Similarity (Evidence vs Answer)")
            
            # Display Generation Metrics (BLEU & ROUGE)
            # [FITUR UTAMA: AKADEMIS METRIC UNTUK CAPTIONING (BLEU & ROUGE)]
            with st.expander("👁️ View Generated Visual Contexts & Metrics", expanded=False):
                st.info("ℹ️ **Captioning Metrics:** Evaluasi seberapa mirip deskripsi AI dengan deskripsi Manusia (Ground Truth).")
                
                for idx, ctx in enumerate(visual_contexts):
                    fname = results[idx]['filename']
                    
                    # 1. Hitung Metrik
                    metrics = {"bleu4": 0.0, "rougeL": 0.0}
                    if fname in gt_captions:
                        references = [c.lower().split() for c in gt_captions[fname]]
                        candidate = ctx.split('\n')[0].lower().split()
                        metrics = calculate_caption_metrics(references, candidate)

                    # 2. Tampilkan Layout
                    c1, c2 = st.columns([3, 1])
                    
                    with c1:
                        st.markdown(f"**Img {idx+1} ({fname}):**")
                        st.text(ctx) # Tampilkan teks caption AI
                        
                    with c2:
                        # --- BLEU-4 METRIC ---
                        # Tentukan warna status BLEU
                        b_score = metrics['bleu4']
                        if b_score > 0.30: b_label = "Excellent 🟢"
                        elif b_score > 0.15: b_label = "Good 🟡"
                        else: b_label = "Low 🔴"
                        
                        st.metric(
                            label="BLEU-4 (Precision)", 
                            value=f"{b_score:.4f}", 
                            help="Mengukur akurasi urutan kata (n-gram). Seberapa tepat kata-kata yang dipilih AI dibandingkan referensi manusia."
                        )
                        st.caption(f"Status: **{b_label}**") # Status khusus BLEU
                        
                        st.divider() # Garis pemisah kecil
                        
                        # --- ROUGE-L METRIC ---
                        # Tentukan warna status ROUGE
                        r_score = metrics['rougeL']
                        if r_score > 0.35: r_label = "Excellent 🟢"
                        elif r_score > 0.20: r_label = "Good 🟡"
                        else: r_label = "Low 🔴"

                        st.metric(
                            label="ROUGE-L (Recall)", 
                            value=f"{r_score:.4f}",
                            help="Mengukur kelengkapan struktur kalimat (LCS). Seberapa lengkap informasi/frasa yang tertangkap dibandingkan referensi manusia."
                        )
                        st.caption(f"Status: **{r_label}**") # Status khusus ROUGE

                    st.divider() # Garis pemisah antar gambar

        st.divider()
        st.caption(f"Total Time: {time.time() - start_time:.2f}s")
        clear_memory()

if __name__ == "__main__":
    main()