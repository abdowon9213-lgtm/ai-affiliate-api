import uuid
import logging
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
from openai import OpenAI
import os

# 1. إعداد السجلات الآمنة
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AffiliateAI")

# 2. جلب رابط Redis وتكوين محدد المعدل (SlowAPI)
REDIS_STORAGE_URI = os.getenv("REDIS_STORAGE_URI", "redis://localhost:6379")

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=REDIS_STORAGE_URI,
    default_limits=["5/minute"]
)

app = FastAPI(
    title="AI Affiliate Marketing API - Fail-Closed",
    version="3.5.0",
    description="FastAPI service with strict Fail-Closed Redis and SlowAPI integration."
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = OpenAI()

# 3. نموذج التحقق Pydantic v2 للمدخلات
class AdRequest(BaseModel):
    product_name: str = Field(..., min_length=2, max_length=100)
    category: str = Field(..., min_length=2, max_length=50)
    key_features: str = Field(..., min_length=5, max_length=500)
    affiliate_link: HttpUrl = Field(...)

# 4. Middleware لتوليد ومعالجة معرف الطلب الفريد
@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.error(f"System processing error [Request-ID: {request_id}]")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id}
        )

# 5. معالج مركزى لأخطاء Redis لتطبيق سياسة (Fail-Closed) حصرياً
@app.exception_handler(RedisError)
async def redis_fail_closed_handler(request: Request, exc: RedisError):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.critical(f"[FAIL-CLOSED ACTIVATED] Redis failure detected [Request-ID: {request_id}] - Error: {type(exc).__name__}")
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Security policy enforcement: Rate limiter backend is unreachable. Request blocked (Fail-Closed).",
            "request_id": request_id
        }
    )

# 6. نقطة النهاية (Endpoint) المؤمنة
@app.post("/api/v1/generate-ads")
@limiter
