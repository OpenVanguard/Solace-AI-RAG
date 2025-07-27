from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime
import pymongo
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import uvicorn
from bson import ObjectId
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the chatbot instance
chatbot_instance = None

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    relevant_documents: List[Dict[str, Any]]
    query: str
    session_id: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    knowledge_base_size: int
    timestamp: str

class StatsResponse(BaseModel):
    total_donations: int
    total_events: int
    total_feedback: int
    total_posts: int
    knowledge_base_size: int

class NGOChatbotAPI:
    def __init__(self, mongo_uri: str, db_name: str, openai_api_key: str):
        """
        Initialize the NGO RAG Chatbot API
        """
        try:
            # MongoDB setup
            self.client = pymongo.MongoClient(mongo_uri)
            self.db = self.client[db_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection successful")
            
            # Collections
            self.donations_collection = self.db.donations
            self.events_collection = self.db.events
            self.feedback_collection = self.db.feedback
            self.posts_collection = self.db.posts
            
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # OpenAI setup
            self.openai_client = OpenAI(api_key=openai_api_key)
            
            # Vector storage
            self.documents = []
            self.embeddings = None
            self.index = None
            
            # Build knowledge base
            self.build_knowledge_base()
            
            logger.info("NGO Chatbot API initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            raise
    
    def extract_donation_info(self, donation: Dict) -> str:
        """Extract relevant information from donation document"""
        info = []
        if 'fullName' in donation:
            info.append(f"Donor: {donation['fullName']}")
        if 'amount' in donation:
            info.append(f"Amount: ₹{donation['amount']}")
        if 'status' in donation:
            info.append(f"Status: {donation['status']}")
        if 'orderId' in donation:
            info.append(f"Order ID: {donation['orderId']}")
        if 'createdAt' in donation:
            try:
                if isinstance(donation['createdAt'], dict) and '$date' in donation['createdAt']:
                    timestamp = donation['createdAt']['$date']['$numberLong']
                    date = datetime.fromtimestamp(int(timestamp) / 1000)
                else:
                    date = donation['createdAt']
                info.append(f"Date: {date.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass
        
        return f"Donation - {', '.join(info)}"
    
    def extract_event_info(self, event: Dict) -> str:
        """Extract relevant information from event document"""
        info = []
        if 'title' in event:
            info.append(f"Title: {event['title']}")
        if 'description' in event:
            info.append(f"Description: {event['description']}")
        if 'date' in event:
            info.append(f"Date: {event['date']}")
        if 'time' in event:
            info.append(f"Time: {event['time']}")
        if 'location' in event:
            info.append(f"Location: {event['location']}")
        if 'participants' in event:
            info.append(f"Participants: {len(event['participants'])}")
        
        return f"Event - {', '.join(info)}"
    
    def extract_feedback_info(self, feedback: Dict) -> str:
        """Extract relevant information from feedback document"""
        message = feedback.get('message', '')
        if 'createdAt' in feedback:
            try:
                if isinstance(feedback['createdAt'], dict) and '$date' in feedback['createdAt']:
                    timestamp = feedback['createdAt']['$date']['$numberLong']
                    date = datetime.fromtimestamp(int(timestamp) / 1000)
                    return f"Feedback - {message} (Date: {date.strftime('%Y-%m-%d %H:%M')})"
            except:
                pass
        return f"Feedback - {message}"
    
    def extract_post_info(self, post: Dict) -> str:
        """Extract relevant information from post document"""
        info = []
        if 'content' in post:
            info.append(f"Content: {post['content']}")
        if 'fullName' in post:
            info.append(f"Author: {post['fullName']}")
        if 'status' in post:
            info.append(f"Status: {post['status']}")
        if 'createdAt' in post:
            try:
                if isinstance(post['createdAt'], dict) and '$date' in post['createdAt']:
                    timestamp = post['createdAt']['$date']['$numberLong']
                    date = datetime.fromtimestamp(int(timestamp) / 1000)
                    info.append(f"Date: {date.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass
        
        return f"Post - {', '.join(info)}"
    
    def build_knowledge_base(self):
        """Build the knowledge base from all collections"""
        logger.info("Building knowledge base...")
        
        try:
            # Extract data from all collections
            donations = list(self.donations_collection.find())
            events = list(self.events_collection.find())
            feedback = list(self.feedback_collection.find())
            posts = list(self.posts_collection.find())
            
            logger.info(f"Found {len(donations)} donations, {len(events)} events, {len(feedback)} feedback, {len(posts)} posts")
            
            # Process documents
            for donation in donations:
                doc_text = self.extract_donation_info(donation)
                doc_id = str(donation.get('_id', donation.get('id', {}).get('$oid', '')))
                self.documents.append({
                    'text': doc_text,
                    'type': 'donation',
                    'id': doc_id,
                    'raw_data': donation
                })
            
            for event in events:
                doc_text = self.extract_event_info(event)
                doc_id = str(event.get('_id', event.get('id', {}).get('$oid', '')))
                self.documents.append({
                    'text': doc_text,
                    'type': 'event',
                    'id': doc_id,
                    'raw_data': event
                })
            
            for fb in feedback:
                doc_text = self.extract_feedback_info(fb)
                doc_id = str(fb.get('_id', fb.get('id', {}).get('$oid', '')))
                self.documents.append({
                    'text': doc_text,
                    'type': 'feedback',
                    'id': doc_id,
                    'raw_data': fb
                })
            
            for post in posts:
                doc_text = self.extract_post_info(post)
                doc_id = str(post.get('_id', post.get('id', {}).get('$oid', '')))
                self.documents.append({
                    'text': doc_text,
                    'type': 'post',
                    'id': doc_id,
                    'raw_data': post
                })
            
            # Create embeddings
            if self.documents:
                texts = [doc['text'] for doc in self.documents]
                logger.info("Creating embeddings...")
                self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
                
                # Build FAISS index
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(self.embeddings)
                self.index.add(self.embeddings.astype('float32'))
                
                logger.info(f"Knowledge base built with {len(self.documents)} documents")
            else:
                logger.warning("No documents found in database")
                
        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}")
            raise
    
    def search_similar_documents(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents using vector similarity"""
        if not self.index or not self.documents:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.documents)))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents) and score > 0.1:  # Minimum relevance threshold
                    results.append({
                        'document': {
                            'text': self.documents[idx]['text'],
                            'type': self.documents[idx]['type'],
                            'id': self.documents[idx]['id']
                        },
                        'score': float(score)
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using OpenAI GPT with retrieved context"""
        try:
            # Prepare context
            context = "\n".join([doc['document']['text'] for doc in context_docs])
            
            system_prompt = """You are a helpful assistant for an NGO/donation website. 
            Use the provided context to answer questions about donations, events, feedback, and posts.
            Be friendly, informative, and focused on helping users understand the organization's work.
            If the context doesn't contain relevant information, say so politely and offer general help.
            Keep responses concise but helpful."""
            
            user_prompt = f"""Context from our database:
{context}

User question: {query}

Please provide a helpful response based on the context above."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request. Please try again."
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function"""
        try:
            # Search for relevant documents
            relevant_docs = self.search_similar_documents(query, k=3)
            
            # Generate response
            response = self.generate_response(query, relevant_docs)
            
            return {
                'response': response,
                'relevant_documents': relevant_docs,
                'query': query
            }
        except Exception as e:
            logger.error(f"Error in chat function: {str(e)}")
            return {
                'response': "I'm sorry, I encountered an error. Please try again.",
                'relevant_documents': [],
                'query': query
            }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get basic statistics about the data"""
        try:
            return {
                'total_donations': self.donations_collection.count_documents({}),
                'total_events': self.events_collection.count_documents({}),
                'total_feedback': self.feedback_collection.count_documents({}),
                'total_posts': self.posts_collection.count_documents({}),
                'knowledge_base_size': len(self.documents)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {
                'total_donations': 0,
                'total_events': 0,
                'total_feedback': 0,
                'total_posts': 0,
                'knowledge_base_size': 0
            }
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy"""
        try:
            self.client.admin.command('ping')
            return True
        except:
            return False

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot_instance
    
    # Load configuration from environment variables
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = os.getenv("DB_NAME", "ngo_database")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable is required")
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    try:
        chatbot_instance = NGOChatbotAPI(MONGO_URI, DB_NAME, OPENAI_API_KEY)
        logger.info("Chatbot instance created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    if chatbot_instance:
        chatbot_instance.client.close()

app = FastAPI(
    title="NGO RAG Chatbot API",
    description="API for NGO donation and social work chatbot using RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "NGO RAG Chatbot API is running!",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    is_healthy = chatbot_instance.is_healthy()
    knowledge_base_size = len(chatbot_instance.documents) if chatbot_instance.documents else 0
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        database_connected=is_healthy,
        knowledge_base_size=knowledge_base_size,
        timestamp=datetime.now().isoformat()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        result = chatbot_instance.chat(request.message)
        
        return ChatResponse(
            response=result['response'],
            relevant_documents=result['relevant_documents'],
            query=result['query'],
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        stats = chatbot_instance.get_statistics()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/refresh")
async def refresh_knowledge_base(background_tasks: BackgroundTasks):
    """Refresh the knowledge base with latest data from database"""
    global chatbot_instance
    
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    def refresh_task():
        try:
            chatbot_instance.build_knowledge_base()
            logger.info("Knowledge base refreshed successfully")
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {str(e)}")
    
    background_tasks.add_task(refresh_task)
    
    return {"message": "Knowledge base refresh initiated"}

if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )