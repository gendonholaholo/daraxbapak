from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from .logging import logger

class AGNOError(Exception):
    """Base exception class for AGNO service"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for the application"""
    if isinstance(exc, AGNOError):
        logger.error(f"AGNO Error: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message}
        )
    elif isinstance(exc, HTTPException):
        logger.error(f"HTTP Error: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    else:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        ) 