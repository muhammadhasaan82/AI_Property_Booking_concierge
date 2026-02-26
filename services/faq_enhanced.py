"""
Enhanced FAQ Service with PDF Processing and Vector Search
Handles company policy questions using semantic search
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# PDF Processing
from PyPDF2 import PdfReader

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Load environment variables
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize ChromaDB path
CHROMA_PATH = Path(__file__).parent.parent / "chroma_faq"
CHROMA_PATH.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FAQService class — replaces global state with dependency injection
# ---------------------------------------------------------------------------

class FAQService:
    """
    Encapsulates vector store, embeddings, and FAQ operations.
    Inject via FastAPI app.state or pass explicitly.
    """

    def __init__(self, chroma_path: Optional[Path] = None, openai_api_key: Optional[str] = None):
        self._openai_api_key = openai_api_key or OPENAI_API_KEY
        self._chroma_path = chroma_path or CHROMA_PATH
        self._chroma_path.mkdir(exist_ok=True)
        self._vector_store: Optional[Chroma] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._healthy = False

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    # --- PDF ingestion ---

    @staticmethod
    def load_pdf_document(pdf_path: str) -> str:
        """Load and extract text from PDF document."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num}]\n{page_text}\n"
            return text
        except Exception as e:
            print(f"[FAQ] Error loading PDF: {e}")
            return ""

    def process_policy_document(self, pdf_path: str, force_reload: bool = False) -> Chroma:
        if not self._openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                openai_api_key=self._openai_api_key,
                model="text-embedding-3-small",
            )

        if self._vector_store is not None and not force_reload:
            return self._vector_store

        persist_directory = str(self._chroma_path)
        if os.path.exists(persist_directory) and not force_reload:
            try:
                self._vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self._embeddings,
                    collection_name="company_policies",
                )
                self._healthy = True
                print("[FAQ] Loaded existing vector store")
                return self._vector_store
            except Exception as e:
                print(f"[FAQ] Error loading existing store: {e}, creating new one")

        print(f"[FAQ] Processing PDF document: {pdf_path}")
        pdf_text = self.load_pdf_document(pdf_path)
        if not pdf_text:
            raise ValueError("Could not extract text from PDF")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            length_function=len, separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_text(pdf_text)
        documents = []
        for i, chunk in enumerate(chunks):
            page_match = re.search(r'\[Page (\d+)\]', chunk)
            page_num = page_match.group(1) if page_match else "Unknown"
            clean_chunk = re.sub(r'\[Page \d+\]', '', chunk).strip()
            doc = Document(
                page_content=clean_chunk,
                metadata={"source": "Company Policy", "page": page_num, "chunk_index": i, "document": "company_policy.pdf"},
            )
            documents.append(doc)
        print(f"[FAQ] Created {len(documents)} document chunks")

        self._vector_store = Chroma.from_documents(
            documents=documents, embedding=self._embeddings,
            persist_directory=persist_directory, collection_name="company_policies",
        )
        self._healthy = True
        print("[FAQ] Vector store created and persisted")
        return self._vector_store

    # --- Semantic search ---

    def semantic_search(self, question: str, k: int = 3, score_threshold: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
        if self._vector_store is None:
            pdf_path = Path(__file__).parent.parent / "Company policy.pdf"
            if not pdf_path.exists():
                return "Company policy document not found. Please ensure the PDF is uploaded.", []
            self.process_policy_document(str(pdf_path))

        results = self._vector_store.similarity_search_with_score(question, k=k)
        relevant_docs = [(doc, score) for doc, score in results if score >= score_threshold]
        if not relevant_docs:
            relevant_docs = [results[0]] if results else []
        if not relevant_docs:
            return "I couldn't find specific information about that in our policies. Would you like to speak with a human agent?", []

        sources: List[Dict[str, Any]] = []
        combined_context = ""
        for doc, score in relevant_docs:
            content = doc.page_content.strip()
            page = doc.metadata.get("page", "Unknown")
            combined_context += content + "\n\n"
            sources.append({"content": content, "page": page, "score": score, "metadata": doc.metadata})

        answer = generate_concise_answer(question, combined_context)
        pages = list(set(doc.metadata.get("page", "Unknown") for doc, _ in relevant_docs))
        if pages and pages != ["Unknown"]:
            page_refs = ", ".join(f"Page {p}" for p in pages if p != "Unknown")
            answer += f"\n\n[Reference: {page_refs} of Company Policy]"
        return answer, sources

    # --- Health & initialization ---

    def initialize(self) -> bool:
        """Initialize FAQ system. Returns True on success."""
        try:
            pdf_path = Path(__file__).parent.parent / "Company policy.pdf"
            if pdf_path.exists():
                self.process_policy_document(str(pdf_path))
                print("[FAQ] System initialized successfully")
                return True
            else:
                print("[FAQ][CRITICAL] Company policy.pdf not found — FAQ will not function")
                return False
        except Exception as e:
            print(f"[FAQ][CRITICAL] Initialization failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Module-level singleton for backwards compatibility
# ---------------------------------------------------------------------------
_faq_service = FAQService()


def load_pdf_document(pdf_path: str) -> str:
    return FAQService.load_pdf_document(pdf_path)


def process_policy_document(pdf_path: str, force_reload: bool = False) -> Chroma:
    return _faq_service.process_policy_document(pdf_path, force_reload)


def semantic_faq_search(question: str, k: int = 3, score_threshold: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
    return _faq_service.semantic_search(question, k, score_threshold)


def detect_faq_intent(user_text: str) -> bool:
    """
    Detect if the user's message is asking about policies, terms, or FAQs.

    Uses NLP-powered semantic classification via nlp_engine instead of
    hardcoded keyword arrays.

    Args:
        user_text: User's input text

    Returns:
        True if this appears to be a FAQ/policy question
    """
    from . import nlp_engine
    return nlp_engine.detect_faq_intent(user_text)


def enhanced_faq_agent(user_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhanced FAQ agent that uses semantic search on policy documents
    
    Args:
        user_text: User's question
        context: Optional context from the conversation state
    
    Returns:
        Dictionary with reply and metadata
    """
    if not user_text or not user_text.strip():
        return {
            "reply": "Please ask me a specific question about our policies or services.",
            "tool_result": {"ok": False, "error": "Empty question"}
        }
    
    try:
        # Perform semantic search
        answer, sources = semantic_faq_search(user_text)
        
        # Check if we found a good answer
        if sources and sources[0]["score"] >= 0.6:
            # High confidence answer
            result = {
                "reply": answer,
                "tool_result": {
                    "ok": True,
                    "sources": sources,
                    "confidence": "high"
                }
            }
        elif sources and sources[0]["score"] >= 0.4:
            # Medium confidence - add disclaimer
            result = {
                "reply": f"{answer}\n\n[Note: If this doesn't fully answer your question, I can connect you with our support team.]",
                "tool_result": {
                    "ok": True,
                    "sources": sources,
                    "confidence": "medium"
                }
            }
        else:
            # Low confidence or no results
            result = {
                "reply": "I couldn't find specific information about that in our policies. Would you like me to:\n1. Try rephrasing your question\n2. Connect you with our support team\n3. Continue with your booking",
                "tool_result": {
                    "ok": False,
                    "confidence": "low",
                    "need_clarification": True
                }
            }
        
        # Add context preservation and continuation prompt if in booking flow
        if context and context.get("in_booking_flow"):
            result["preserve_context"] = True
            result["return_to"] = context.get("return_to", "booking")
            # Add continuation prompt
            if result["tool_result"].get("ok"):
                result["reply"] += "\n\nWould you like to know something else about our policies, or shall we continue with your booking?"
        
        return result
        
    except Exception as e:
        print(f"Error in FAQ agent: {e}")
        return {
            "reply": "I'm having trouble accessing the policy information right now. Please try again or contact support directly.",
            "tool_result": {
                "ok": False,
                "error": str(e)
            }
        }


# Initialize the vector store on module load
def initialize_faq_system():
    """Initialize the FAQ system with the company policy document."""
    return _faq_service.initialize()


def generate_concise_answer(question: str, context: str) -> str:
    """
    Generate a concise, summarized answer using OpenAI
    
    Args:
        question: User's question
        context: Retrieved policy text
    
    Returns:
        Concise answer (5-20 lines based on complexity)
    """
    if not OPENAI_API_KEY:
        # Fallback to extracting key sentences if no OpenAI key
        return extract_key_sentences(context, question)
    
    try:
        import httpx
        
        # Determine question complexity using VADER + heuristics
        from . import nlp_engine
        vader = nlp_engine._get_vader()
        q_lower = question.lower()
        q_words = len(q_lower.split())
        scores = vader.polarity_scores(q_lower)
        # Simple: short, direct questions; Complex: longer, analytical questions
        is_simple = q_words <= 6 and "?" in question
        is_complex = q_words > 10 or any(w in q_lower for w in ["explain", "how does", "process", "procedure"])
        
        # Set length guidance
        if is_simple and not is_complex:
            length_guide = "Provide a brief, direct answer in 3-5 lines."
        elif is_complex:
            length_guide = "Provide a comprehensive but concise answer in 10-15 lines, covering key points."
        else:
            length_guide = "Provide a clear, concise answer in 5-10 lines."
        
        system_prompt = f"""You are a helpful property rental assistant. Answer questions based ONLY on the provided policy text.
{length_guide}
Be specific and direct. Use bullet points for multiple items.
Do not add information not present in the context."""
        
        user_prompt = f"""Based on the following policy text, answer this question concisely:

Question: {question}

Policy Text:
{context[:2000]}  # Limit context to avoid token limits

Provide a clear, direct answer:"""
        
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            return answer
        else:
            print(f"OpenAI API error: {response.status_code}")
            return extract_key_sentences(context, question)
            
    except Exception as e:
        print(f"Error generating concise answer: {e}")
        return extract_key_sentences(context, question)


def extract_key_sentences(context: str, question: str, max_lines: int = 10) -> str:
    """
    Fallback method to extract key sentences when OpenAI is not available
    
    Args:
        context: Full policy text
        question: User's question
        max_lines: Maximum number of lines to return
    
    Returns:
        Key sentences related to the question
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', context)
    
    # Find sentences with keywords from the question
    question_words = set(question.lower().split())
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Count matching words
        matches = sum(1 for word in question_words if word in sentence_lower)
        if matches >= 2 or any(kw in sentence_lower for kw in ["refund", "cancel", "pet", "deposit", "check"]):
            relevant_sentences.append(sentence.strip())
    
    # Return the most relevant sentences
    result = " ".join(relevant_sentences[:5])
    
    # Clean up the text
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'[©\[\]]', '', result)
    
    # Limit to reasonable length
    if len(result) > 800:
        result = result[:800] + "..."
    
    return result


# Optional: Auto-initialize when module is imported
# Uncomment the line below if you want automatic initialization
# initialize_faq_system()

