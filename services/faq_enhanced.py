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

# Global vector store instance
_vector_store: Optional[Chroma] = None
_embeddings: Optional[OpenAIEmbeddings] = None


def load_pdf_document(pdf_path: str) -> str:
    """Load and extract text from PDF document"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                # Add page number for reference
                text += f"\n[Page {page_num}]\n{page_text}\n"
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""


def process_policy_document(pdf_path: str, force_reload: bool = False) -> Chroma:
    """
    Process the company policy PDF and create/load vector store
    
    Args:
        pdf_path: Path to the PDF document
        force_reload: If True, recreate the vector store even if it exists
    
    Returns:
        Chroma vector store instance
    """
    global _vector_store, _embeddings
    
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found in environment variables")
    
    # Initialize embeddings if not already done
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
    
    # Check if vector store already exists and we're not forcing reload
    if _vector_store is not None and not force_reload:
        return _vector_store
    
    # Check if persisted store exists
    persist_directory = str(CHROMA_PATH)
    if os.path.exists(persist_directory) and not force_reload:
        try:
            _vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=_embeddings,
                collection_name="company_policies"
            )
            print("Loaded existing vector store")
            return _vector_store
        except Exception as e:
            print(f"Error loading existing store: {e}, creating new one")
    
    # Load and process the PDF
    print(f"Processing PDF document: {pdf_path}")
    pdf_text = load_pdf_document(pdf_path)
    
    if not pdf_text:
        raise ValueError("Could not extract text from PDF")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create documents with metadata
    chunks = text_splitter.split_text(pdf_text)
    documents = []
    
    for i, chunk in enumerate(chunks):
        # Extract page number if present
        page_match = re.search(r'\[Page (\d+)\]', chunk)
        page_num = page_match.group(1) if page_match else "Unknown"
        
        # Clean the chunk text
        clean_chunk = re.sub(r'\[Page \d+\]', '', chunk).strip()
        
        # Create document with metadata
        doc = Document(
            page_content=clean_chunk,
            metadata={
                "source": "Company Policy",
                "page": page_num,
                "chunk_index": i,
                "document": "company_policy.pdf"
            }
        )
        documents.append(doc)
    
    print(f"Created {len(documents)} document chunks")
    
    # Create and persist vector store
    _vector_store = Chroma.from_documents(
        documents=documents,
        embedding=_embeddings,
        persist_directory=persist_directory,
        collection_name="company_policies"
    )
    
    print("Vector store created and persisted")
    return _vector_store


def semantic_faq_search(
    question: str,
    k: int = 3,
    score_threshold: float = 0.5
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Perform semantic search on the FAQ/policy documents
    
    Args:
        question: User's question
        k: Number of relevant chunks to retrieve
        score_threshold: Minimum similarity score threshold
    
    Returns:
        Tuple of (formatted answer, list of source documents)
    """
    global _vector_store
    
    if _vector_store is None:
        # Initialize vector store with company policy PDF
        pdf_path = Path(__file__).parent.parent / "Company policy.pdf"
        if not pdf_path.exists():
            return "Company policy document not found. Please ensure the PDF is uploaded.", []
        
        _vector_store = process_policy_document(str(pdf_path))
    
    # Perform similarity search with scores
    results = _vector_store.similarity_search_with_score(question, k=k)
    
    # Filter by score threshold
    relevant_docs = [(doc, score) for doc, score in results if score >= score_threshold]
    
    if not relevant_docs:
        # If no high-confidence matches, return top result anyway
        if results:
            relevant_docs = [results[0]]
        else:
            return "I couldn't find specific information about that in our policies. Would you like to speak with a human agent?", []
    
    # Collect source content and metadata
    sources = []
    combined_context = ""
    
    for doc, score in relevant_docs:
        content = doc.page_content.strip()
        page = doc.metadata.get("page", "Unknown")
        combined_context += content + "\n\n"
        
        # Track sources
        sources.append({
            "content": content,
            "page": page,
            "score": score,
            "metadata": doc.metadata
        })
    
    # Use OpenAI to generate a concise answer if available
    answer = generate_concise_answer(question, combined_context)
    
    # Add source reference if we have page numbers
    pages = list(set(doc.metadata.get("page", "Unknown") for doc, _ in relevant_docs))
    if pages and pages != ["Unknown"]:
        page_refs = ", ".join(f"Page {p}" for p in pages if p != "Unknown")
        answer += f"\n\n[Reference: {page_refs} of Company Policy]"
    
    return answer, sources


def detect_faq_intent(user_text: str) -> bool:
    """
    Detect if the user's message is asking about policies, terms, or FAQs
    
    Args:
        user_text: User's input text
    
    Returns:
        True if this appears to be a FAQ/policy question
    """
    text_lower = user_text.lower()
    
    # FAQ trigger keywords
    faq_keywords = [
        # Policy-related
        "policy", "policies", "terms", "conditions", "rules", "regulations",
        "requirements", "guidelines", "procedures", "protocol",
        
        # Refund and cancellation
        "refund", "cancel", "cancellation", "reschedule", "postpone",
        "money back", "reimbursement", "compensation", "return",
        
        # Payment
        "payment", "pay", "deposit", "fee", "charge", "cost", "price",
        "billing", "invoice", "transaction", "method of payment",
        
        # Check-in/out
        "check-in", "checkin", "check in", "check-out", "checkout", "check out",
        "arrival", "departure", "early check", "late check",
        
        # Disputes and issues
        "dispute", "complaint", "issue", "problem", "concern", "grievance",
        "resolution", "support", "help", "assistance",
        
        # Amenities and services
        "amenities", "facilities", "services", "included", "provided",
        "wifi", "parking", "breakfast", "cleaning", "maintenance",
        
        # Pets and smoking
        "pet", "pets", "animal", "dog", "cat", "smoking", "smoke",
        
        # Damage and security
        "damage", "security", "deposit", "liability", "responsible",
        "insurance", "coverage", "protection",
        
        # Guest-related
        "guest", "visitor", "occupancy", "capacity", "maximum",
        "additional", "extra person", "children", "infant",
        
        # Question indicators
        "what is", "what are", "what's", "how does", "how do", "how to",
        "can i", "can we", "is it", "are there", "do you", "does the",
        "tell me about", "explain", "clarify", "understand"
    ]
    
    # Check for FAQ keywords
    for keyword in faq_keywords:
        if keyword in text_lower:
            # Additional context check - make sure it's actually a question
            # and not just mentioning these terms in booking context
            question_patterns = [
                r'\?',  # Has question mark
                r'^(what|how|when|where|why|can|do|does|is|are|tell|explain)',  # Starts with question word
                r'(please )?(tell|explain|clarify|help|show) (me |us )?',  # Request patterns
            ]
            
            for pattern in question_patterns:
                if re.search(pattern, text_lower):
                    return True
            
            # Even without question patterns, strong policy keywords should trigger FAQ
            strong_triggers = ["policy", "refund", "cancel", "terms", "conditions", "dispute"]
            if any(trigger in text_lower for trigger in strong_triggers):
                return True
    
    return False


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
    """Initialize the FAQ system with the company policy document"""
    try:
        pdf_path = Path(__file__).parent.parent / "Company policy.pdf"
        if pdf_path.exists():
            process_policy_document(str(pdf_path))
            print("FAQ system initialized successfully")
        else:
            print("Warning: Company policy.pdf not found")
    except Exception as e:
        print(f"Error initializing FAQ system: {e}")


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
        
        # Determine if this is a simple or complex question
        simple_keywords = ["allowed", "permitted", "can i", "is it", "are there", "what time", "when"]
        complex_keywords = ["explain", "how does", "process", "procedure", "policy", "terms", "conditions"]
        
        is_simple = any(kw in question.lower() for kw in simple_keywords)
        is_complex = any(kw in question.lower() for kw in complex_keywords)
        
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

