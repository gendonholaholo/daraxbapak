from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, Dict, List
from .core.config import settings
from .core.logging import logger
from .core.errors import global_exception_handler, AGNOError
from .core.security import get_current_user, authenticate_user, create_access_token, get_current_active_user
from .agents import agent_factory
from datetime import timedelta

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add global exception handler
app.add_exception_handler(Exception, global_exception_handler)

# Request/Response Models
class InterviewRequest(BaseModel):
    session_id: str
    message: str
    topic: Optional[str] = None
    question: Optional[str] = None
    point: Optional[str] = None

class InterviewResponse(BaseModel):
    response: str
    context: List[Dict]
    prompt_type: str

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get(f"{settings.API_V1_STR}/protected")
async def protected_route(current_user = Depends(get_current_user)):
    """Example protected route"""
    return {"message": "This is a protected route", "user": current_user}

@app.post(f"{settings.API_V1_STR}/interview", response_model=InterviewResponse)
async def interview(request: InterviewRequest):
    """Interview endpoint"""
    try:
        agent = agent_factory.get_agent("interviewer")
        response = await agent.process(request.dict())
        return response
    except AGNOError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Interview error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Application shutdown complete")

@app.post(f"{settings.API_V1_STR}/auth/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to get access token"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get(f"{settings.API_V1_STR}/users/me")
async def read_users_me(current_user = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 