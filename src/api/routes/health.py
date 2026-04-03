from fastapi import APIRouter
from src.api.models import HealthResponse

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Healthcheck endpoint.
    """
    return HealthResponse(status="ok", version="1.0")