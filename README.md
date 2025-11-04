# AI Estate - Real Estate Chatbot

A sophisticated AI-powered real estate chatbot built with FastAPI backend and React frontend, designed to provide comprehensive real estate advice and property management assistance.

## 🏠 Features

- **Intelligent Chat Interface**: Natural language processing for real estate queries
- **Property Search & Recommendations**: AI-powered property discovery and filtering
- **Booking Management**: Complete booking workflow with payment integration
- **FAQ System**: Automated responses to common real estate questions
- **Multi-language Support**: English, Arabic, Korean, and Urdu
- **Voice Integration**: OpenAI Realtime API for voice conversations
- **Modern UI**: Beautiful dark-themed interface with Tailwind CSS

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangGraph** - AI agent orchestration
- **ChromaDB** - Vector database for embeddings
- **OpenAI GPT** - Language model integration
- **PostgreSQL** - Database for bookings and user data
- **Supabase** - Backend-as-a-Service

### Frontend
- **React 18** - Modern React with hooks
- **Next.js 14** - Full-stack React framework
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Heroicons** - Beautiful SVG icons

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
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

Once the server is running, you can access:

1. **AI Estate Text Chatbot**:
   - Open http://127.0.0.1:8000/chatbot in your browser
   - Features:
     - Real-time text chat with AI
     - Property search and recommendations
     - Click the phone icon to switch to calling agent

2. **Voice Calling Agent**:
   - Open http://127.0.0.1:8000/static/index.html in your browser
   - Features:
     - Voice conversations with AI
     - Real-time transcription
     - Multi-language support

### 4. Frontend Development Setup (Optional)
```bash
# Navigate to frontend directory
cd Frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
Calling-Agent-Chatbot/
├── Frontend/                 # React Next.js frontend
│   ├── src/app/             # Next.js app directory
│   ├── package.json         # Node.js dependencies
│   └── tailwind.config.ts   # Tailwind configuration
├── public/                   # Static files (legacy)
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

## 🎨 Frontend Development

### Available Scripts
```bash
# Development
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
```

### UI Components
- **Chat Interface**: Real-time messaging with AI
- **Property Cards**: Beautiful property listings
- **Booking Forms**: Streamlined booking process
- **Responsive Design**: Mobile-first approach

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

### Frontend Deployment
```bash
# Build for production
cd Frontend
npm run build

# Deploy to Vercel/Netlify
npm run deploy
```

## 🧪 Testing

```bash
# Backend tests
python -m pytest tests/

# Frontend tests
cd Frontend
npm test

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