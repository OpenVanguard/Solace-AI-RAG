# Solace AI RAG 🤖💙

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://www.mongodb.com/atlas)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-orange.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Intelligent RAG-powered chatbot for NGOs and social organizations to provide instant, context-aware assistance about donations, events, feedback, and community activities.**

Solace AI RAG transforms your organization's data into an intelligent conversational assistant that helps visitors get instant answers about your work, impact, and ways to contribute.

## 🌟 Features

### 🧠 **Intelligent RAG System**
- **Vector Search**: Semantic similarity using sentence transformers
- **Context-Aware Responses**: GPT-powered responses based on your actual data
- **Multi-Source Integration**: Donations, events, feedback, and posts in one system
- **Real-time Updates**: Automatic knowledge base refresh when data changes

### 🚀 **Production-Ready API**
- **RESTful Endpoints**: Easy integration with any website or application
- **High Performance**: FAISS vector database for lightning-fast searches
- **Scalable Architecture**: Handles multiple concurrent users
- **Comprehensive Monitoring**: Health checks, analytics, and performance metrics

### 🛡️ **Enterprise Security**
- **API Authentication**: Secure API key-based access control
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Input Validation**: Protects against malicious inputs
- **CORS Support**: Configurable cross-origin resource sharing

### 📊 **Analytics & Insights**
- **Usage Tracking**: Monitor chat interactions and user engagement
- **Performance Metrics**: Response times and system health monitoring
- **Data Statistics**: Real-time insights into your organization's data

## 🎯 Use Cases

- **Donor Support**: "How can I donate?" "What projects need funding?"
- **Event Information**: "What events are happening this month?" "Where is the next meetup?"
- **Impact Stories**: "What has our organization achieved?" "Show me recent feedback"
- **Volunteer Coordination**: "How can I help?" "What volunteer opportunities are available?"

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   MongoDB       │
│   (React)       │◄──►│   RAG Server     │◄──►│   Atlas         │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   OpenAI GPT     │
                       │   + Vector DB    │
                       │   (FAISS)        │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- MongoDB Atlas account
- OpenAI API key

### 1. Clone Repository

```bash
git clone https://github.com/OpenVanguard/Solace-AI-RAG.git
cd Solace-AI-RAG
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `.env` file:

```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
DB_NAME=your_ngo_database
OPENAI_API_KEY=sk-your-openai-api-key-here
API_SECRET_KEY=your-secret-api-key
ENVIRONMENT=development
```

### 4. Run the Server

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

Your API will be available at `http://localhost:8000`

### 5. Test the API

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message":"How many donations have we received?"}'
```

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send message to chatbot |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Database statistics |
| `POST` | `/refresh` | Refresh knowledge base |

### Chat Endpoint

**Request:**
```json
{
    "message": "How can I donate to your organization?",
    "user_id": "user123",
    "session_id": "session456"
}
```

**Response:**
```json
{
    "response": "You can donate through our website...",
    "relevant_documents": [
        {
            "document": {
                "text": "Donation - Donor: John Doe, Amount: ₹500",
                "type": "donation",
                "id": "doc123"
            },
            "score": 0.95
        }
    ],
    "query": "How can I donate to your organization?",
    "session_id": "session456",
    "timestamp": "2025-01-27T10:30:00"
}
```

## 🔧 Integration Examples

### JavaScript/Web

```javascript
async function askChatbot(message) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    
    const data = await response.json();
    return data.response;
}
```

### React

```jsx
const [response, setResponse] = useState('');

const sendMessage = async (message) => {
    const result = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    
    const data = await result.json();
    setResponse(data.response);
};
```

### Python

```python
import requests

def chat_with_bot(message):
    response = requests.post('http://localhost:8000/chat', 
        json={'message': message}
    )
    return response.json()['response']
```

## 📁 Project Structure

```
Solace-AI-RAG/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── README.md               # This file       
└── tests/                  # Test files
    ├── test_api.py         # API tests
    └── test_chatbot.py     # Chatbot tests
```

## 🐳 Docker Deployment

### 🔧 Build & Run

#### Option 1: Using Docker Compose

```bash
# Build and run
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

#### Option 2: Manual Docker Build

```bash
# Build image
docker build -t solace-ai-rag .

# Run container
docker run -d \
    --name solace-ai-rag \
    -p 8000:8000 \
    --env-file .env \
    solace-ai-rag
```

The API will be available at [http://localhost:8000](http://localhost:8000)

## ☁️ Cloud Deployment - DigitalOcean/AWS/GCP

See [deployment guide](docs/deployment.md) for detailed instructions.

## 📊 Database Schema

Your MongoDB collections should follow this structure:

### Donations Collection
```json
{
    "_id": "ObjectId",
    "fullName": "Donor Name",
    "amount": 1000,
    "status": "completed",
    "orderId": "order_123",
    "createdAt": "ISO Date"
}
```

### Events Collection
```json
{
    "_id": "ObjectId",
    "title": "Event Title",
    "description": "Event Description",
    "date": "2025-05-27",
    "time": "19:29",
    "location": "City Name",
    "participants": [],
    "createdAt": "ISO Date"
}
```

### Feedback Collection
```json
{
    "_id": "ObjectId",
    "message": "User feedback message",
    "createdAt": "ISO Date"
}
```

### Posts Collection
```json
{
    "_id": "ObjectId",
    "content": "Post content",
    "fullName": "Author Name",
    "status": "published",
    "createdAt": "ISO Date"
}
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest

# Run with coverage
pytest --cov=main --cov-report=html
```

## 📈 Performance

- **Response Time**: < 2 seconds average
- **Concurrent Users**: 100+ simultaneous connections
- **Throughput**: 1000+ requests/minute
- **Knowledge Base**: Supports 10K+ documents efficiently

## 🔒 Security

- ✅ API key authentication
- ✅ Rate limiting (10 requests/minute)
- ✅ Input validation and sanitization
- ✅ CORS protection
- ✅ SQL injection prevention
- ✅ XSS protection

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/OpenVanguard/Solace-AI-RAG?style=social)
![GitHub forks](https://img.shields.io/github/forks/OpenVanguard/Solace-AI-RAG?style=social)
![GitHub issues](https://img.shields.io/github/issues/OpenVanguard/Solace-AI-RAG)
![GitHub pull requests](https://img.shields.io/github/issues-pr/OpenVanguard/Solace-AI-RAG)

---

<div align="center">

**Built with ❤️ by [OpenVanguard](https://github.com/OpenVanguard)**

*Empowering NGOs with AI-driven solutions for better community engagement*

[⭐ Star this repo](https://github.com/OpenVanguard/Solace-AI-RAG) | [🐛 Report Bug](https://github.com/OpenVanguard/Solace-AI-RAG/issues) | [💡 Request Feature](https://github.com/OpenVanguard/Solace-AI-RAG/issues)

</div>