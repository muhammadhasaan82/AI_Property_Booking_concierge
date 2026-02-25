# AI Estate - Real Estate Chatbot

A sophisticated AI-powered real estate chatbot built with FastAPI backend and React frontend, designed to provide comprehensive real estate advice and property management assistance.

## 🏠 Features

- **Intelligent Chat Interface**: Natural language processing for real estate queries
- **Property Search & Recommendations**: AI-powered property discovery and filtering
- **Booking Management**: Complete booking workflow with payment integration
- **FAQ System**: Automated responses to common real estate questions
- **Multi-language Support**: English, Arabic, Korean, and Urdu

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangGraph** - AI agent orchestration
- **ChromaDB** - Vector database for embeddings
- **OpenAI GPT** - Language model integration
- **PostgreSQL** - Database for bookings and user data
- **Supabase** - Backend-as-a-Service



## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- PostgreSQL (or use Supabase)

### 1. Clone the Repository
```bash
git clone https://github.com/wenawa/ai_booking.git
cd ai_booking
```

### 2. Backend Setup
```bash
# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the backend server
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Access the Chatbot Interfaces

Once the server is running, you can access the API at:
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
Calling-Agent-Chatbot/
├── services/                 # Core business logic
│   ├── agents.py           # AI agent implementations
│   ├── graph.py            # LangGraph workflow
│   ├── retrieval.py        # Vector search
│   └── dataset_loader.py   # Data ingestion
├── route/                   # API endpoints
│   ├── chat.py             # Chat API
│   ├── booking.py          # Booking management
│   └── health.py           # Health checks
├── main.py                 # FastAPI application
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
SUPABASE_DB_URL=your_supabase_connection_string
SUPABASE_DB_HOST=127.0.0.1
SUPABASE_DB_PORT=54322
SUPABASE_DB_NAME=postgres
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=postgres

# Vector Database
CHROMA_DIR=./chroma
EMBED_MODEL=thenlper/gte-small
CHROMA_COLLECTION=properties

# Dataset Configuration
DATASET_PATH=./services/dataset.csv
DATASET_FORMAT=csv
```

## 📊 Data Sources

The system supports multiple data formats:

- **CSV Files**: Property listings with automatic ingestion
- **Excel Files**: XLSX support for property data
- **JSON Files**: Structured property data
- **PDF Documents**: Policy documents and FAQs
- **Database**: Direct PostgreSQL integration

### Adding New Data Sources
```python
# Example: Load from Excel
python services/dataset_loader.py --path "data/properties.xlsx" --fmt xlsx

# Example: Load from JSON
python services/dataset_loader.py --path "data/properties.json" --fmt json
```

## 🤖 AI Agents

The system includes specialized AI agents:

- **Triage Agent**: Routes user intents to appropriate handlers
- **FAQ Agent**: Answers common real estate questions
- **Property Agent**: Handles property search and recommendations
- **Booking Agent**: Manages booking workflows
- **Status Agent**: Tracks booking status
- **Payment Agent**: Handles payment processing



## 🔌 API Endpoints

### Chat API
```http
POST /api/v1/chat/message
Content-Type: application/json

{
  "message": "I'm looking for a 2-bedroom apartment in New York",
  "filters": {
    "city": "New York",
    "bedrooms": 2
  }
}
```

### Health Check
```http
GET /api/v1/health
```

### Property Search
```http
GET /api/v1/properties?city=New York&bedrooms=2&max_price=3000
```

## 🚀 Deployment

### Backend Deployment
```bash
# Using Docker
docker build -t ai-estate-backend .
docker run -p 8000:8000 ai-estate-backend

# Using Railway/Heroku
git push heroku main
```



## 🧪 Testing

```bash
# Backend tests
python -m pytest tests/



# API smoke tests
python tests/api_smoke.py
```

## 📈 Performance

- **Vector Search**: Sub-second property retrieval
- **Caching**: Redis integration for improved performance
- **CDN**: Static asset optimization
- **Database**: Optimized queries with proper indexing

## 🔒 Security

- **CORS**: Configured for production origins
- **API Keys**: Secure environment variable management
- **Input Validation**: Pydantic models for data validation
- **Rate Limiting**: Protection against abuse

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint for API documentation
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

## 🎯 Roadmap

- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Multi-tenant support
- [ ] Integration with popular real estate platforms
- [ ] Advanced AI features (image recognition, market analysis)

---

**Built with ❤️ for the real estate industry**