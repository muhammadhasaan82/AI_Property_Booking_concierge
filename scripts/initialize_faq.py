"""
Initialize the FAQ system by processing the Company Policy PDF
This script should be run once to set up the vector store
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)

from app.services.faq_enhanced import process_policy_document, initialize_faq_system


def main():
    """Initialize the FAQ system with the company policy document"""
    print("\n" + "="*60)
    print("FAQ System Initialization")
    print("="*60)
    
    # Check if PDF exists
    pdf_path = Path(__file__).resolve().parents[1] / "data" / "Company policy.pdf"
    
    if not pdf_path.exists():
        print(f"\n[ERROR] Company policy.pdf not found at {pdf_path}")
        print("   Please ensure the PDF file is in the data/ directory")
        return
    
    print(f"\n[PDF] Found PDF: {pdf_path}")
    print("[Processing] Processing document and creating vector store...")
    
    try:
        # Process the PDF and create vector store
        vector_store = process_policy_document(str(pdf_path), force_reload=True)
        
        print("\n[SUCCESS] FAQ system initialized with the following details:")
        print(f"   - PDF processed: Company policy.pdf")
        print(f"   - Vector store location: {Path(__file__).resolve().parents[1] / 'data' / 'chroma_faq'}")
        print("   - Embeddings model: BAAI/bge-small-en-v1.5")
        
        # Test with a sample question
        print("\n[TEST] Testing with sample question...")
        from app.services.faq_enhanced import semantic_faq_search
        
        test_question = "What is the refund policy?"
        answer, sources = semantic_faq_search(test_question, k=1)
        
        if answer:
            print(f"   Question: {test_question}")
            print(f"   Answer preview: {answer[:150]}...")
            print("\n[SUCCESS] FAQ system is working correctly!")
        else:
            print("[WARNING] No answer found for test question")
        
    except Exception as e:
        print(f"\n[ERROR] Error during initialization: {e}")
        print("   Please check your embedding dependencies and internet connection")
        return
    
    print("\n" + "="*60)
    print("Initialization Complete!")
    print("="*60)
    print("\nThe FAQ agent is now ready to answer questions about company policies.")
    print("You can test it using: python test_faq_agent.py")


if __name__ == "__main__":
    main()

