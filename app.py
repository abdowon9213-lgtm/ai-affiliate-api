import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="AI Affiliate Marketing API")

# قراءة الإعدادات من Render
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL")

# لا نستخدم مفتاحًا وهميًا
client = None
if GROQ_API_KEY:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

class AdRequest(BaseModel):
    product_name: str = Field(..., min_length=1)
    features: str = Field(..., min_length=1)
    target_audience: str = Field(..., min_length=1)
    language: str = Field(...)

LANG_INSTRUCTIONS = {
    "ar_msa": "Modern Standard Arabic.",
    "ar_ly": "Natural Libyan Arabic dialect.",
    "en": "English."
}

@app.get("/", response_class=HTMLResponse)
def read_root():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "index.html")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="index.html not found"
        )

@app.post("/api/v1/generate-ads")
def generate_ads(payload: AdRequest):

    if not GROQ_API_KEY or client is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on Render"
        )

    if not GROQ_MODEL:
        raise HTTPException(
            status_code=503,
            detail="GROQ_MODEL is not configured on Render"
        )

    lang_instruction = LANG_INSTRUCTIONS.get(payload.language)

    if not lang_instruction:
        raise HTTPException(
            status_code=422,
            detail="Unsupported language. Use ar_ly, ar_msa, or en."
        )

    system_prompt = (
        "You are a professional marketing copywriter.\n"
        "Use only the product information provided. "
        "Never invent specifications, prices, discounts, guarantees, "
        "or results.\n"
        f"Write exclusively in: {lang_instruction}\n"
        "Create a persuasive, clear advertisement for the target audience."
    )

    user_content = (
        f"Product: {payload.product_name}\n"
        f"Features: {payload.features}\n"
        f"Target audience: {payload.target_audience}"
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.4,
            max_tokens=600
        )

        ad = response.choices[0].message.content

        if not ad:
            raise HTTPException(
                status_code=502,
                detail="The AI model returned an empty response"
            )

        return {"success": True, "ad": ad}

    except HTTPException:
        raise

    except Exception:
        import logging
        logging.exception("Groq advertisement generation failed")

        raise HTTPException(
            status_code=502,
            detail="تعذر توليد الإعلان من خدمة الذكاء الاصطناعي. راجع Render Logs."
        )

@app.get("/models")
def list_models():

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured"
        )

    try:
        models = client.models.list()
        return {
            "models": [model.id for model in models.data]
        }

    except Exception:
        import logging
        logging.exception("Failed to retrieve Groq models")

        raise HTTPException(
            status_code=502,
            detail="Could not retrieve models from Groq"
        )
