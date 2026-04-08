from fastapi import APIRouter
from src.api.models import HealthResponse

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint to verify the service is running.
    
    Returns:
        HealthResponse with service status and version.
    """
    return HealthResponse(status="ok", version="1.0")