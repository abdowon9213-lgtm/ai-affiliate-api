import os
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("affiliate-app")

app = FastAPI(
    title="Saudi Affiliate Generator",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

client = OpenAI(
    api_key=GROQ_API_KEY or "missing-key",
    base_url="https://api.groq.com/openai/v1"
)


class GenerateRequest(BaseModel):
    product_name: str = Field(min_length=2, max_length=160)
    features: str = Field(min_length=5, max_length=1200)
    target_audience: str = "المتسوقون في السعودية"
    product_url: str | None = None
    affiliate_url: str = Field(min_length=8, max_length=2000)
    extra_notes: str | None = None


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/v1/generate-ads")
def generate_ads(payload: GenerateRequest):

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="إعداد مفتاح Groq غير مكتمل في Render."
        )

    disclosure = (
        "إعلان بعمولة: قد أحصل على عمولة عند "
        "الشراء عبر هذا الرابط، دون تكلفة إضافية عليك."
    )

    prompt = f"""
اكتب 3 إعلانات مختلفة باللهجة السعودية.

اسم المنتج: {payload.product_name}
المميزات: {payload.features}
الجمهور: {payload.target_audience}
ملاحظات: {payload.extra_notes or "لا توجد"}

التعليمات:
- استخدم اللهجة السعودية الطبيعية.
- لا تخترع أسعاراً أو خصومات أو تقييمات.
- لا تضف معلومات غير موجودة.
- اجعل الإعلانات مناسبة لفيسبوك.
- لا تضع الرابط داخل نص الإعلان.
- أعد JSON فقط بهذا الشكل:

{{
  "variants": [
    {{"title":"عنوان","copy":"نص الإعلان","hashtags":["وسم1","وسم2"]}},
    {{"title":"عنوان","copy":"نص الإعلان","hashtags":["وسم1","وسم2"]}},
    {{"title":"عنوان","copy":"نص الإعلان","hashtags":["وسم1","وسم2"]}}
  ]
}}
"""

    try:
        result = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.7,
            max_tokens=1400,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "أنت كاتب إعلانات متخصص باللهجة السعودية. "
                        "التزم بالمعلومات المقدمة فقط."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = result.choices[0].message.content
        data = json.loads(content or "{}")
        variants = data.get("variants", [])

        if len(variants) != 3:
            raise ValueError("Invalid response")

        for item in variants:
            if not all(
                key in item
                for key in ["title", "copy", "hashtags"]
            ):
                raise ValueError("Missing fields")

    except Exception:
        logger.exception("Groq generation failed")
        raise HTTPException(
            status_code=502,
            detail="تعذر إنشاء الإعلانات حالياً. حاول مرة أخرى."
        )

    return {
        "success": True,
        "country": "السعودية",
        "dialect": "اللهجة السعودية",
        "disclosure": disclosure,
        "affiliate_url": payload.affiliate_url,
        "variants": variants
}
