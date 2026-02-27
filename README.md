# AI Hotel Booking Platform (Hybrid Python + Rust)

A sophisticated AI-powered hotel booking chatbot and platform built with a hybrid **Python + Rust** architecture. The system combines the conversational capabilities of LLMs and LangGraph in Python with the blazing fast computational performance of Rust microservices, all communicating via the custom serialized **TOON** protocol.

## 🏠 Features

- **Intelligent Chat Interface**: A ChatGPT-like interactive UI built with Chainlit.
- **Natural Language Processing**: Dynamic intent triage, sentiment analysis, entity extraction using VADER, spaCy, and sentence-transformers.
- **High-Performance Rust Microservices**: Deterministic computation offloaded to an Axum-based Rust Autonomous Gateway.
- **Custom TOON Protocol**: High-efficiency, LLM-optimized data serialization format for Python ↔ Rust communication.
- **Property Search & Recommendations**: AI-powered property discovery and filtering from a rich dataset.
- **Booking Management**: Complete booking validation workflow, pricing calculation, and fraud checks via Rust.
- **FAQ System**: Automated semantic responses to common hospitality questions.

## 🛠️ Tech Stack

### Python Orchestrator & UI
- **Chainlit** - Conversational ChatGPT-like frontend interface.
- **FastAPI** - Modern Python web framework for background API routing.
- **LangGraph** - AI agent orchestration and state graph.
- **spaCy, VADER, sentence-transformers** - Local NLP evaluation models.
- **OpenAI GPT** - Structured LLM processing.

### Rust Autonomous Gateway
- **Rust (Cargo)** - High-performance systems language backend.
- **Axum & Tokio** - Async HTTP gateway and event loops.
- **Serde** - Robust value serialization.
- **Custom Tool Registry** - Modular Search, Validation, Pricing, Sentiment, and Fraud tools.

### Database
- **PostgreSQL / Supabase** - Database for persisting bookings and users.
- **ChromaDB** - Vector database for embeddings.

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **Rust toolchain** (cargo, rustc - MSVC on Windows)
- **PostgreSQL / Supabase** account

### 1. Clone the Repository
```bash
git clone https://github.com/muhammadhasaan82/Hotel_Booking.git
cd Hotel_Booking
```

### 2. Rust Gateway Setup
Start the high-performance Rust microservices server locally. This handles heavy computations like search, booking validation, and sentiment analysis.

```bash
cd rust_gateway
cargo update
# Build and run the Axum server
cargo run --release
```
The Rust gateway will run continuously on `http://localhost:3001`.

### 3. Python Backend & UI Setup
In a separate terminal, install the dependencies and boot the python orchestrator and Chainlit interface.

```bash
# Return to the root directory
cd ..

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download required spaCy core model
python -m spacy download en_core_web_sm

# Set up environment variables
cp services/env.example .env
# Edit .env with your OpenAI API keys and Supabase credentials
```

### 4. Access the Application
Start the Chainlit conversational interface:

```bash
chainlit run chainlit_app.py -w
```
The application will open in your browser at `http://localhost:8000`.

## 📁 Project Structure

```text
Hotel_Booking/
├── chainlit_app.py         # Conversational UI entrypoint
├── services/               # Python Core Business Logic
│   ├── agents.py           # LangGraph agent implementations
│   ├── graph.py            # Chat state graph
│   ├── nlp_engine.py       # Entity extraction and semantic classification
│   ├── rust_client.py      # Async client for Rust gateway calls
│   └── toon.py             # Custom TOON protocol serializer
├── rust_gateway/           # Rust Microservice Backend
│   ├── src/
│   │   ├── main.rs         # Axum server & dual-format parsing
│   │   ├── gateway.rs      # Gateway routing logic
│   │   ├── toon.rs         # Native Rust TOON implementation
│   │   └── tools/          # Tool definitions (search, sentiment, validator)
│   └── Cargo.toml          # Rust dependencies
├── route/                  # FastAPI endpoints
├── main.py                 # FastAPI Application
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration

### Environment Variables
Edit your `.env` file referencing `services/env.example`:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration (Supabase)
SUPABASE_DB_URL=...
SUPABASE_DB_PASSWORD=...

# Rust Gateway
RUST_GATEWAY_URL=http://localhost:3001
RUST_TIMEOUT=5.0
```

## 🤖 AI Agents & Tools

The platform routes user intents dynamically using local NLP models before executing tasks:

- **Triage Agent**: Routes user intents across the state graph.
- **FAQ Agent**: Answers common hotel/real-estate questions.
- **Property Agent**: Connects to the **Rust `PropertySearchTool`** to semantically filter local properties via the TOON protocol.
- **Booking Agent**: Validates parameters instantly using the **Rust `BookingValidatorTool`** before writing data limits into Postgres.
- **Status & Payment**: Automatically generates tracking links seamlessly in the UI.

## 🛑 Testing
Both halves of the stack carry their own rigorous testing:

```bash
# Test Python NLP, TOON format, and Langchain integrations
python -m pytest tests/test_nlp_engine.py -v
python -m pytest tests/test_toon.py -v

# Test Rust TOON protocols natively
cd rust_gateway
cargo test
```

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.