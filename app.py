import os
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from openai import OpenAI

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Fail-Closed AI Affiliate Marketing API", version="3.5.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("CRITICAL: OPENAI_API_KEY environment variable is missing!")

client = OpenAI(api_key=openai_api_key)

@app.get("/")
def read_root():
    return {"status": "online", "mode": "Fail-Closed AI Affiliate API"}

@app.post("/api/v1/generate-ads")
@limiter.limit("5/minute")
async def generate_ads(request: Request):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Generate an affiliate marketing headline for a smartphone."}],
            max_tokens=50
        )
        return {
            "success": True,
            "headline": response.choices[0].message.content
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

