#!/usr/bin/env python3
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import os
import uuid
import requests
from datetime import datetime
from pathlib import Path
import tempfile
import time
import numpy as np
import urllib3
import google.generativeai as genai
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials as fb_credentials
from firebase_admin import firestore

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from project root .env (override any pre-set vars)
try:
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
except Exception:
    # Fallback to default loader
    load_dotenv(override=True)

# Import all components
import fitz  # PyMuPDF
import faiss

# Set availability flags
PDF_AVAILABLE = True
AI_AVAILABLE = True

app = FastAPI(
    title="HackRx API with FAISS & Gemini",
    description="Insurance Policy AI Query System with PDF Processing, FAISS Vector Search, and Google Gemini Integration",
    version="1.0.0"
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev convenience: allow all origins for local file:// and localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Security
security = HTTPBearer()

# Pydantic models
class QueryRequest(BaseModel):
    documents: str  # URL to PDF document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Configuration
API_KEY = os.getenv("HACKRX_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Gemini primary
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
FIREBASE_COLLECTION_DOCS = os.getenv("FIREBASE_COLLECTION_DOCS", "docs")
FIREBASE_COLLECTION_CHUNKS = os.getenv("FIREBASE_COLLECTION_CHUNKS", "chunks")

# Validate required environment variables
if not API_KEY or not API_KEY.strip():
    raise RuntimeError("HACKRX_API_KEY not set. Please create a .env with HACKRX_API_KEY and restart the server.")

if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
    raise RuntimeError("GEMINI_API_KEY not set. Please create a .env with GEMINI_API_KEY and restart the server.")

# Configure Gemini client if key is present
try:
    if GEMINI_API_KEY and GEMINI_API_KEY.strip():
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as _e:
    # Non-fatal; health endpoint will reflect connectivity
    pass

# Global variables for AI components
faiss_index = None
document_chunks = []  # Store chunks in memory
document_embeddings = []  # Store embeddings in memory
document_cache = {}  # Cache for processed documents
embedding_cache = {}  # Cache for embeddings
last_document_id = None  # Track the most recently uploaded/processed document
chunk_doc_ids = []  # Parallel to document_chunks for mapping back to doc

# Firestore
firestore_client = None
firestore_connected = False

def init_firestore():
    global firestore_client, firestore_connected
    try:
        if GOOGLE_APPLICATION_CREDENTIALS and Path(GOOGLE_APPLICATION_CREDENTIALS).exists():
            cred = fb_credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {
                    'projectId': FIREBASE_PROJECT_ID
                })
        else:
            # Try default credentials (gcloud env)
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
        firestore_client = firestore.client()
        firestore_connected = True
    except Exception as e:
        print(f"⚠️  Firestore init failed: {e}")
        firestore_client = None
        firestore_connected = False

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def initialize_ai_components():
    """Initialize FAISS index and verify Gemini configuration"""
    global faiss_index
    
    if not AI_AVAILABLE:
        print("⚠️  AI components not available")
        return False
    
    try:
        # Initialize FAISS index with ultra-fast keyword hashing dimension
        dimension = 64  # Ultra-fast keyword hashing dimension
        faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        if GEMINI_API_KEY and GEMINI_API_KEY.strip():
            print("✅ AI components initialized successfully (Gemini ready)!")
            return True
        else:
            print("⚠️  GEMINI_API_KEY not set. Gemini features will be disabled.")
            return False
        
    except Exception as e:
        print(f"❌ Error initializing AI components: {e}")
        return False

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and save to temporary file"""
    try:
        # Disable SSL verification for problematic URLs
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file using PyMuPDF"""
    try:
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        
        # Clean up temporary file
        os.unlink(file_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        
        return text
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """Split text into semantically coherent chunks optimized for insurance documents"""
    if not text.strip():
        return []
    
    # Clean and normalize text
    text = text.replace('\r', '\n').replace('\t', ' ')
    lines = text.split('\n')
    
    chunks = []
    current_chunk = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new section (common in insurance documents)
        section_indicators = [
            'SECTION', 'CLAUSE', 'ARTICLE', 'PART', 'CHAPTER',
            'DEFINITIONS', 'EXCLUSIONS', 'COVERAGE', 'LIMITS',
            'CONDITIONS', 'ENDORSEMENTS', 'SCHEDULE', 'POLICY'
        ]
        
        is_section_start = any(indicator in line.upper() for indicator in section_indicators)
        
        # If adding this line would exceed chunk size, save current chunk
        if current_chunk and (len(current_chunk + line) > chunk_size or is_section_start):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += " " + line
            else:
                current_chunk = line
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks (likely headers or footers)
    chunks = [chunk for chunk in chunks if len(chunk) > 50]
    
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate ultra-fast keyword-based embeddings for insurance documents"""
    if not texts:
        return []
    
    # Comprehensive insurance terminology for keyword hashing
    insurance_terms = {
        # Health Insurance
        'health': ['health', 'medical', 'hospital', 'doctor', 'treatment', 'surgery', 'medication', 'prescription'],
        'coverage': ['coverage', 'insured', 'policyholder', 'beneficiary', 'claim', 'premium', 'deductible'],
        'disease': ['disease', 'illness', 'condition', 'diagnosis', 'symptoms', 'chronic', 'acute'],
        'waiting': ['waiting', 'period', 'exclusion', 'pre-existing', 'grace', 'renewal'],
        
        # Life Insurance
        'life': ['life', 'death', 'mortality', 'survival', 'term', 'whole', 'universal'],
        'benefit': ['benefit', 'sum', 'assured', 'death', 'maturity', 'surrender'],
        
        # Motor Insurance
        'motor': ['motor', 'vehicle', 'car', 'auto', 'accident', 'collision', 'comprehensive'],
        'damage': ['damage', 'repair', 'replacement', 'liability', 'third', 'party'],
        
        # Property Insurance
        'property': ['property', 'building', 'house', 'fire', 'theft', 'burglary', 'natural'],
        'structure': ['structure', 'contents', 'belongings', 'furniture', 'appliances'],
        
        # Travel Insurance
        'travel': ['travel', 'trip', 'journey', 'overseas', 'international', 'domestic'],
        'emergency': ['emergency', 'evacuation', 'repatriation', 'medical', 'assistance'],
        
        # Liability Insurance
        'liability': ['liability', 'negligence', 'damages', 'compensation', 'legal'],
        'professional': ['professional', 'malpractice', 'errors', 'omissions'],
        
        # Marine Insurance
        'marine': ['marine', 'cargo', 'ship', 'vessel', 'freight', 'transit'],
        'shipping': ['shipping', 'transport', 'logistics', 'warehouse'],
        
        # Financial Insurance
        'financial': ['financial', 'credit', 'bond', 'guarantee', 'fidelity'],
        'investment': ['investment', 'fund', 'portfolio', 'market', 'risk'],
        
        # Policy Terms
        'policy': ['policy', 'contract', 'agreement', 'terms', 'conditions'],
        'exclusion': ['exclusion', 'limitation', 'restriction', 'exception'],
        'claim': ['claim', 'notification', 'settlement', 'investigation'],
        
        # Time-based Terms
        'time': ['time', 'period', 'duration', 'term', 'renewal', 'expiry'],
        'date': ['date', 'effective', 'commencement', 'termination'],
        
        # Medical Terms
        'medical': ['medical', 'hospitalization', 'surgery', 'diagnosis', 'treatment'],
        'medication': ['medication', 'drug', 'prescription', 'pharmacy'],
        
        # Coverage Types
        'coverage_type': ['individual', 'family', 'group', 'corporate', 'comprehensive'],
        'limit': ['limit', 'maximum', 'minimum', 'sub-limit', 'ceiling']
    }
    
    embeddings = []
    
    for text in texts:
        # Create a 64-dimensional vector based on keyword presence
        vector = [0.0] * 64
        
        text_lower = text.lower()
        
        # Hash keywords into vector positions
        for category, keywords in insurance_terms.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Simple hash function to map keyword to vector position
                    hash_val = hash(keyword) % 64
                    vector[hash_val] += 1.0
        
        # Normalize the vector
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x/magnitude for x in vector]
        
        embeddings.append(vector)
    
    return embeddings

def generate_answer_with_gemini(question: str, context: str) -> str:
    """Generate answer using Google Gemini with optimized prompt for insurance accuracy"""
    if not AI_AVAILABLE or not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
        return f"AI service not available. Please check if GEMINI_API_KEY is set in environment variables. Question: {question}"
    try:
        # Build prompt similar to other providers for consistency
        prompt = f"""You are an expert insurance policy analyst with deep knowledge of health insurance policies. Your task is to provide accurate, precise answers based solely on the provided policy document context.

POLICY DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based EXCLUSIVELY on the information provided in the policy document context above
2. If the specific information is not found in the context, respond with: "The provided policy document does not contain specific information about [exact topic]"
3. Be precise and include exact details, numbers, and terms from the policy when available
4. Use clear, professional language appropriate for insurance documentation
5. If you find relevant information, quote it accurately from the policy
6. Do not make assumptions or provide information not present in the context
7. If the context is insufficient, clearly state what specific information is missing
8. For numerical values (periods, amounts, percentages), be exact
9. For policy terms and conditions, be specific about requirements and limitations
10. Structure your answer logically with clear points

ANSWER:"""

        # Prefer a fast, cost-effective model (configurable via GEMINI_MODEL)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        # Handle potential safety blocks or empty responses
        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            # Some SDK versions return candidates
            candidates = getattr(response, "candidates", []) or []
            if candidates and getattr(candidates[0], "content", None):
                parts = getattr(candidates[0].content, "parts", []) or []
                answer = " ".join([getattr(p, "text", "") for p in parts]).strip()
        if not answer:
            return f"The provided policy document does not contain specific information about this question: {question}"
        if len(answer) > 1200:
            sentences = answer.split('. ')
            truncated = ""
            for s in sentences:
                if len(truncated + s + '. ') <= 1200:
                    truncated += s + '. '
                else:
                    break
            answer = truncated.strip()
        return answer
    except Exception as e:
        # Help debug common model errors
        return f"Error generating answer: {str(e)}. Please verify GEMINI_MODEL='{GEMINI_MODEL}' is available to your key. Question: {question}"

def store_chunks_in_faiss(chunks: List[str], embeddings: List[List[float]], document_id: str):
    """Store document chunks and embeddings in FAISS index"""
    global faiss_index, document_chunks, document_embeddings, chunk_doc_ids
    
    if not chunks or not embeddings:
        return
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Add to FAISS index
    faiss_index.add(embeddings_array)
    
    # Store chunks and embeddings in memory
    start_idx = len(document_chunks)
    document_chunks.extend(chunks)
    document_embeddings.extend(embeddings)
    chunk_doc_ids.extend([document_id] * len(chunks))
    
    print(f"✅ Stored {len(chunks)} chunks for document {document_id}")

def search_relevant_chunks(question: str, top_k: int = 8) -> List[str]:
    """Search for relevant chunks using hybrid approach (FAISS + keyword matching)"""
    if not document_chunks or faiss_index is None:
        return []
    
    # Comprehensive insurance terms for keyword matching
    insurance_terms = {
        'health': ['health', 'medical', 'hospital', 'doctor', 'treatment', 'surgery', 'medication', 'prescription', 'diagnosis', 'symptoms'],
        'life': ['life', 'death', 'mortality', 'survival', 'term', 'whole', 'universal', 'benefit', 'sum assured'],
        'motor': ['motor', 'vehicle', 'car', 'auto', 'accident', 'collision', 'comprehensive', 'damage', 'repair'],
        'property': ['property', 'building', 'house', 'fire', 'theft', 'burglary', 'natural', 'disaster'],
        'travel': ['travel', 'trip', 'journey', 'overseas', 'international', 'domestic', 'emergency'],
        'liability': ['liability', 'negligence', 'damages', 'compensation', 'legal', 'professional'],
        'marine': ['marine', 'cargo', 'ship', 'vessel', 'freight', 'transit', 'shipping'],
        'financial': ['financial', 'credit', 'bond', 'guarantee', 'fidelity', 'investment'],
        'policy': ['policy', 'contract', 'agreement', 'terms', 'conditions', 'coverage'],
        'claim': ['claim', 'notification', 'settlement', 'investigation', 'benefit'],
        'exclusion': ['exclusion', 'limitation', 'restriction', 'exception', 'waiting', 'period'],
        'premium': ['premium', 'payment', 'grace', 'renewal', 'expiry', 'lapse'],
        'hospital': ['hospital', 'hospitalization', 'admission', 'discharge', 'room', 'icu'],
        'surgery': ['surgery', 'surgical', 'operation', 'procedure', 'anesthesia'],
        'medication': ['medication', 'drug', 'prescription', 'pharmacy', 'medicine'],
        'diagnosis': ['diagnosis', 'diagnostic', 'test', 'laboratory', 'pathology'],
        'treatment': ['treatment', 'therapy', 'rehabilitation', 'physiotherapy'],
        'emergency': ['emergency', 'urgent', 'critical', 'ambulance', 'evacuation'],
        'preventive': ['preventive', 'checkup', 'vaccination', 'screening', 'wellness'],
        'maternity': ['maternity', 'pregnancy', 'delivery', 'childbirth', 'antenatal'],
        'dental': ['dental', 'tooth', 'oral', 'dental surgery', 'orthodontics'],
        'ophthalmic': ['ophthalmic', 'eye', 'vision', 'glasses', 'contact lens', 'cataract'],
        'mental': ['mental', 'psychiatric', 'psychological', 'counseling', 'therapy'],
        'rehabilitation': ['rehabilitation', 'physiotherapy', 'occupational', 'speech therapy'],
        'prosthesis': ['prosthesis', 'artificial', 'limb', 'wheelchair', 'crutches'],
        'organ': ['organ', 'transplant', 'donor', 'recipient', 'tissue'],
        'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 'yoga'],
        'alternative': ['alternative', 'complementary', 'traditional', 'herbal'],
        'daycare': ['daycare', 'day care', 'ambulatory', 'outpatient', 'same day'],
        'dialysis': ['dialysis', 'kidney', 'renal', 'hemodialysis', 'peritoneal'],
        'chemotherapy': ['chemotherapy', 'radiation', 'oncology', 'cancer', 'tumor'],
        'vaccination': ['vaccination', 'immunization', 'vaccine', 'inoculation'],
        'health_check': ['health check', 'checkup', 'screening', 'preventive', 'wellness'],
        'ncd': ['ncd', 'no claim', 'discount', 'bonus', 'loading'],
        'portability': ['portability', 'transfer', 'switch', 'migration'],
        'pre_existing': ['pre-existing', 'pre existing', 'existing', 'condition'],
        'grace_period': ['grace period', 'grace', 'payment', 'premium'],
        'waiting_period': ['waiting period', 'waiting', 'exclusion period'],
        'sub_limit': ['sub limit', 'sub-limit', 'limit', 'ceiling', 'maximum'],
        'room_rent': ['room rent', 'room', 'accommodation', 'boarding'],
        'icu': ['icu', 'intensive care', 'critical care', 'ccu'],
        'copay': ['copay', 'co-pay', 'co-payment', 'deductible', 'excess'],
        'network': ['network', 'provider', 'hospital', 'doctor', 'panel'],
        'cashless': ['cashless', 'cash less', 'direct settlement'],
        'reimbursement': ['reimbursement', 'reimburse', 'claim', 'settlement']
    }
    
    # Get question embedding
    question_embedding = get_embeddings([question])[0]
    question_embedding_array = np.array([question_embedding], dtype=np.float32)
    
    # FAISS search
    if faiss_index.ntotal > 0:
        scores, indices = faiss_index.search(question_embedding_array, min(top_k * 2, faiss_index.ntotal))
        faiss_results = [(document_chunks[i], scores[0][j]) for j, i in enumerate(indices[0]) if i < len(document_chunks)]
    else:
        faiss_results = []
    
    # Keyword matching
    question_lower = question.lower()
    keyword_matches = []
    
    for i, chunk in enumerate(document_chunks):
        chunk_lower = chunk.lower()
        score = 0
        
        for category, keywords in insurance_terms.items():
            for keyword in keywords:
                if keyword in question_lower and keyword in chunk_lower:
                    score += 2  # Higher weight for exact matches
                elif keyword in chunk_lower:
                    score += 1  # Lower weight for chunk relevance
        
        if score > 0:
            keyword_matches.append((chunk, score))
    
    # Combine and rank results
    all_results = {}
    
    # Add FAISS results
    for chunk, score in faiss_results:
        all_results[chunk] = all_results.get(chunk, 0) + score * 0.7  # Weight FAISS results
    
    # Add keyword results
    for chunk, score in keyword_matches:
        all_results[chunk] = all_results.get(chunk, 0) + score * 0.3  # Weight keyword results
    
    # Sort by score and return top results
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    relevant_chunks = [chunk for chunk, score in sorted_results[:top_k]]
    
    return relevant_chunks

 

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting HackRx API Server with FAISS & Gemini...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📄 Main Endpoint: POST http://localhost:8000/hackrx/run")
    print("🔐 Authentication: Bearer token required")
    print("=" * 60)
    initialize_ai_components()
    # Initialize Firestore and load persisted chunks (if any)
    init_firestore()
    if firestore_connected and firestore_client is not None:
        try:
            print("☁️  Loading chunks from Firestore (this may take a moment)...")
            docs_ref = firestore_client.collection(FIREBASE_COLLECTION_DOCS)
            chunks_ref = firestore_client.collection(FIREBASE_COLLECTION_CHUNKS)
            # Stream chunks and rebuild index
            all_chunks = []
            all_embeddings = []
            all_doc_ids = []
            for ch in chunks_ref.stream():
                data = ch.to_dict() or {}
                text = data.get('text', '')
                doc_id = data.get('doc_id', '')
                if text and doc_id:
                    all_chunks.append(text)
                    all_doc_ids.append(doc_id)
            if all_chunks:
                embs = get_embeddings(all_chunks)
                store_chunks_in_faiss(all_chunks, embs, document_id="__bulk__load__")
                # Overwrite mapping with true doc_ids for each chunk
                global chunk_doc_ids
                chunk_doc_ids[-len(all_doc_ids):] = all_doc_ids
                print(f"✅ Loaded {len(all_chunks)} chunks from Firestore")
        except Exception as e:
            print(f"⚠️  Failed to load Firestore chunks: {e}")

@app.get("/")
async def root():
    return {
        "message": "HackRx API with FAISS & Gemini Integration",
        "version": "1.0.0",
        "status": "running",
        "ai_provider": "Google Gemini",
        "vector_search": "FAISS",
        "endpoints": {
            "health": "/health",
            "main": "POST /hackrx/run",
            "ask": "POST /ask-policy",
            "upload": "POST /upload-policy",
            "ingest_url": "POST /ingest-url",
            "docs": "GET /kb/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_available": AI_AVAILABLE,
        "pdf_processing": PDF_AVAILABLE,
        "faiss_connected": faiss_index is not None,
        "gemini_connected": GEMINI_API_KEY is not None and GEMINI_API_KEY.strip() != "",
        "firestore_connected": firestore_connected,
        "chunks_stored": len(document_chunks)
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Main endpoint for processing insurance policy queries"""
    start_time = time.time()
    
    try:
        # Check if document is already cached
        document_id = request.documents
        if document_id in document_cache:
            print(f"📄 Using cached document: {document_id}")
            chunks = document_cache[document_id]
        else:
            print(f"📄 Processing new document: {document_id}")
            
            # Download and extract text from PDF
            pdf_path = download_pdf_from_url(request.documents)
            text = extract_text_from_pdf(pdf_path)
            
            # Chunk the text
            chunks = chunk_text(text)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No meaningful content extracted from PDF")
            
            # Cache the chunks
            document_cache[document_id] = chunks
            print(f"✅ Extracted {len(chunks)} chunks from PDF")
        
        # Generate embeddings and store in FAISS
        embeddings = get_embeddings(chunks)
        store_chunks_in_faiss(chunks, embeddings, document_id)
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            print(f"🤔 Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            # Search for relevant chunks
            relevant_chunks = search_relevant_chunks(question)
            
            if not relevant_chunks:
                answers.append(f"The provided policy document does not contain specific information about this question: {question}")
                continue
            
            # Combine relevant chunks into context
            context = "\n\n".join(relevant_chunks)
            
            # Generate answer using AI (Gemini only)
            try:
                answer = generate_answer_with_gemini(question, context)
            except Exception as e:
                print(f"❌ Gemini error: {e}")
                answer = f"Error generating answer: {str(e)}. Question: {question}"
            
            answers.append(answer)
            
            print(f"✅ Generated answer for question {i+1}")
        
        processing_time = time.time() - start_time
        print(f"🎉 Processed {len(request.questions)} questions in {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ---------- Chat-style helper endpoints ----------

class AskRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None

class AskResponse(BaseModel):
    answer: str

@app.post("/ask-policy", response_model=AskResponse)
async def ask_policy(payload: AskRequest, api_key: str = Depends(verify_api_key)):
    """Ask a question against the currently indexed document in memory"""
    if not document_chunks or faiss_index is None or faiss_index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No document indexed yet. Please upload a policy first.")

    question = payload.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    relevant = search_relevant_chunks(question)
    # Optional filter by doc_ids
    if payload.doc_ids:
        filtered = []
        for ch in relevant:
            try:
                idx = document_chunks.index(ch)
                if chunk_doc_ids[idx] in payload.doc_ids:
                    filtered.append(ch)
            except ValueError:
                continue
        relevant = filtered
    if not relevant:
        return AskResponse(answer="The provided policy document does not contain specific information about this question.")

    context = "\n\n".join(relevant)
    answer = generate_answer_with_gemini(question, context)
    return AskResponse(answer=answer)

@app.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """Upload a PDF or TXT policy, extract text, chunk, embed and index it. Returns a brief summary."""
    global last_document_id
    try:
        filename = file.filename or "uploaded"
        content = await file.read()

        text = ""
        # Handle TXT directly
        if file.content_type in ("text/plain",) or filename.lower().endswith(".txt"):
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                text = content.decode("latin-1", errors="ignore")
        else:
            # Assume PDF or binary -> write temp and use extractor
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            text = extract_text_from_pdf(tmp_path)

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from uploaded file")

        # Chunk and embed
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful content extracted from file")

        embeddings = get_embeddings(chunks)
        doc_id = f"upload:{uuid.uuid4()}"
        store_chunks_in_faiss(chunks, embeddings, doc_id)
        document_cache[doc_id] = chunks
        last_document_id = doc_id

        # Make a short summary using first few chunks
        summary_context = "\n\n".join(chunks[:5])
        summary_prompt = "Provide a concise summary of the key coverage, limits, and exclusions in this policy."
        summary = generate_answer_with_gemini(summary_prompt, summary_context)

        # Persist to Firestore
        if firestore_connected and firestore_client is not None:
            try:
                firestore_client.collection(FIREBASE_COLLECTION_DOCS).document(doc_id).set({
                    'id': doc_id,
                    'title': filename,
                    'source': filename,
                    'type': 'pdf' if filename.lower().endswith('.pdf') else 'txt',
                    'chunks_count': len(chunks),
                    'created_at': datetime.now().isoformat()
                })
                batch = firestore_client.batch()
                chunks_col = firestore_client.collection(FIREBASE_COLLECTION_CHUNKS)
                for i, ch_text in enumerate(chunks):
                    doc_ref = chunks_col.document(f"{doc_id}:{i}")
                    batch.set(doc_ref, {
                        'doc_id': doc_id,
                        'chunk_id': i,
                        'text': ch_text
                    })
                batch.commit()
            except Exception as e:
                print(f"⚠️  Firestore persist failed: {e}")

        return {"document_id": doc_id, "chunks_indexed": len(chunks), "summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")

class IngestUrlRequest(BaseModel):
    url: str
    title: Optional[str] = None

@app.post("/ingest-url")
async def ingest_url(payload: IngestUrlRequest, api_key: str = Depends(verify_api_key)):
    """Ingest a PDF/TXT by URL, persist to Firestore, and index locally"""
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        # Download and extract
        pdf_path = download_pdf_from_url(url)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful content extracted from URL")
        embeddings = get_embeddings(chunks)
        doc_id = f"url:{uuid.uuid4()}"
        store_chunks_in_faiss(chunks, embeddings, doc_id)
        document_cache[doc_id] = chunks

        # Persist to Firestore
        if firestore_connected and firestore_client is not None:
            try:
                firestore_client.collection(FIREBASE_COLLECTION_DOCS).document(doc_id).set({
                    'id': doc_id,
                    'title': payload.title or url.split('/')[-1] or 'Document',
                    'source': url,
                    'type': 'pdf',
                    'chunks_count': len(chunks),
                    'created_at': datetime.now().isoformat()
                })
                batch = firestore_client.batch()
                chunks_col = firestore_client.collection(FIREBASE_COLLECTION_CHUNKS)
                for i, ch_text in enumerate(chunks):
                    doc_ref = chunks_col.document(f"{doc_id}:{i}")
                    batch.set(doc_ref, {
                        'doc_id': doc_id,
                        'chunk_id': i,
                        'text': ch_text
                    })
                batch.commit()
            except Exception as e:
                print(f"⚠️  Firestore persist failed: {e}")

        # Quick summary
        summary_context = "\n\n".join(chunks[:5])
        summary_prompt = "Provide a concise summary of the key coverage, limits, and exclusions in this policy."
        summary = generate_answer_with_gemini(summary_prompt, summary_context)
        return {"document_id": doc_id, "chunks_indexed": len(chunks), "summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ingest URL error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest URL: {str(e)}")

@app.get("/kb/docs")
async def list_docs(api_key: str = Depends(verify_api_key)):
    if not (firestore_connected and firestore_client is not None):
        return {"docs": []}
    try:
        results = []
        for d in firestore_client.collection(FIREBASE_COLLECTION_DOCS).stream():
            data = d.to_dict() or {}
            results.append(data)
        return {"docs": results}
    except Exception as e:
        print(f"⚠️  list_docs failed: {e}")
        return {"docs": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
