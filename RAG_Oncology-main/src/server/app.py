from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
import traceback
from datetime import datetime
from sqlalchemy.orm import Session

from src.agent_workflow.workflow import WorkFlow

# Import database configuration and models
from src.config.database import get_db, Base, engine
from src.models.user_memory import UserMemory, init_db
from src.helpers.user_memory_manager import UserMemoryManager

# Initialize the database
init_db()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cancer Agent API")

# Add middleware for request logging
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise

app.middleware('http')(log_requests)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize any required services here

class ChatMessage(BaseModel):
    message: str
    patient_id: Optional[int] = 0

class ChatResponse(BaseModel):
    response: str
    confidence: Optional[float] = None
    source: Optional[str] = None

# User Memory Models
class UserMemoryBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    user_id: int

class UserMemoryCreate(UserMemoryBase):
    pass

class UserMemoryUpdate(UserMemoryBase):
    name: Optional[str] = None
    description: Optional[str] = None
    user_id: Optional[int] = None  # Optional for updates

class UserMemoryResponse(UserMemoryBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True  # Updated from orm_mode for Pydantic v2

@app.get("/")
async def root():
    return {"message": "Cancer Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# User Memory Endpoints
@app.post("/user-memories/", response_model=UserMemoryResponse, status_code=status.HTTP_201_CREATED)
def create_user_memory(user_memory: UserMemoryCreate, db: Session = Depends(get_db)):
    """
    Create a new user memory
    
    Note: Each user can only have one memory entry.
    """
    try:
        return UserMemoryManager.create_memory(
            user_id=user_memory.user_id,
            name=user_memory.name,
            description=user_memory.description
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user-memories/user/{user_id}", response_model=UserMemoryResponse)
def read_user_memory_by_user(user_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific user memory by user ID"""
    memory = UserMemoryManager.get_memory_by_user(user_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"No memory found for user {user_id}")
    return memory

@app.put("/user-memories/user/{user_id}", response_model=UserMemoryResponse)
def update_user_memory_by_user(
    user_id: int, 
    user_memory: UserMemoryUpdate, 
    db: Session = Depends(get_db)
):
    """Update a user memory by user ID"""
    try:
        update_data = user_memory.dict(exclude_unset=True)
        # Remove user_id from update data if it's None to avoid overwriting
        if 'user_id' in update_data and update_data['user_id'] is None:
            del update_data['user_id']
            
        updated = UserMemoryManager.update_memory(
            user_id=user_id,
            name=update_data.get('name'),
            description=update_data.get('description')
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"No memory found for user {user_id}")
        return updated
    except Exception as e:
        logger.error(f"Error updating user memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/user-memories/user/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_memory_by_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user memory by user ID"""
    if not UserMemoryManager.delete_memory(user_id):
        raise HTTPException(status_code=404, detail=f"No memory found for user {user_id}")
    return None

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process chat messages and return responses
    
    Args:
        message: ChatMessage containing the message and optional patient_id
        
    Returns:
        ChatResponse with the AI's response
    """
    try:
        work_flow = WorkFlow()
        # Pass both message and patient_id to the workflow
        response = work_flow(
            message=message.message,
            patient_id=message.patient_id if hasattr(message, 'patient_id') else 0
        )
        
        # Get the last message from the workflow response
        messages = response.get('messages', [])
        if not messages:
            raise HTTPException(status_code=500, detail="No response generated")
            
        ai_response = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        logger.info(f"AI response for patient {getattr(message, 'patient_id', 0)}: {ai_response}")
        
        # Return the response in the expected format
        return ChatResponse(
            response=ai_response,
            confidence=1.0,  # Default confidence
            source="cancer_agent"  # Default source
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))