from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai
import os

nltk.download('punkt')

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Configure Google Gemini API
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Replace with your API Key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# ✅ Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Load the embedding model
MODEL_NAME = "intfloat/multilingual-e5-large"
embedding_model = SentenceTransformer(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF, using OCR if necessary."""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text += text + "\n"
        else:
            images = page.get_pixmap()
            img_bytes = io.BytesIO(images.tobytes("png"))
            img = Image.open(img_bytes)
            text = pytesseract.image_to_string(img, lang="mal")
            full_text += text + "\n"

    return full_text.strip()

def build_bm25_index(text_chunks):
    """Build BM25 index for keyword-based search."""
    tokenized_texts = [word_tokenize(chunk.lower()) for chunk in text_chunks]
    return BM25Okapi(tokenized_texts)

def retrieve_answer_bm25(query, text_chunks, bm25_index):
    """Retrieve answer using BM25."""
    tokenized_query = word_tokenize(query.lower())
    scores = bm25_index.get_scores(tokenized_query)
    best_match_idx = scores.argmax()
    return text_chunks[best_match_idx] if scores[best_match_idx] > 0 else None

def generate_answer_gemini(query, context):
    """Generate answer using Gemini AI."""
    try:
        response = gemini_model.generate_content(f"പ്രശ്നം: {query}\n\nപാഠഭാഗം: {context}\n\nഉത്തരം മലയാളത്തിൽ:")
        return response.text if response.text.strip() else "⚠️ ഉത്തരം ലഭ്യമല്ല."
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

@app.post("/upload/")
async def process_pdf(file: UploadFile = File(...), query: str = Form(...)):
    """Receive PDF and query, then return the answer."""
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    text = extract_text_from_pdf(file_path)
    os.remove(file_path)  # Delete after processing

    if not text:
        return {"answer": "⚠️ No text extracted from the PDF."}

    # Split text into chunks
    text_chunks = text.split("\n\n")
    bm25_index = build_bm25_index(text_chunks)

    # Retrieve the best matching answer
    raw_answer = retrieve_answer_bm25(query, text_chunks, bm25_index)
    
    if raw_answer:
        gemini_answer = generate_answer_gemini(query, raw_answer)
        return {"answer": gemini_answer}
    else:
        return {"answer": "⚠️ ഉത്തരം ലഭ്യമല്ല."}
