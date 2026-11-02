import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="المسوق الذكي 2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    product_name: str
    features: str
    target_audience: str = ""
    product_url: str = ""
    affiliate_url: str
    extra_notes: str = ""

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/generate-ads")
def generate_ads(data: GenerateRequest):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="إعداد مفتاح الخدمة غير مكتمل في Render."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    prompt = f"""
أنت كاتب إعلانات تسويقية محترف متخصص في السوق السعودي.

اكتب ثلاثة إعلانات مختلفة للمنتج التالي:

اسم المنتج: {data.product_name}
مميزات المنتج: {data.features}
الجمهور المستهدف: {data.target_audience}
ملاحظات إضافية: {data.extra_notes}

قواعد إلزامية:
1. اكتب بالعربية فقط وبلهجة سعودية طبيعية ومفهومة.
2. ممنوع إدخال أي كلمات إنجليزية أو فرنسية.
3. ممنوع الأخطاء الإملائية والعبارات الركيكة.
4. لا تخترع مواصفات أو فوائد غير موجودة في بيانات المنتج.
5. لا تبالغ ولا تقدم وعوداً غير مثبتة.
6. اجعل كل إعلان مختلفاً في الأسلوب والعنوان.
7. أضف مجموعة هاشتاقات عربية مناسبة لكل إعلان.
8. أضف إفصاح العمولة بالنص التالي:
"إعلان بعمولة: قد أحصل على عمولة عند الشراء عبر هذا الرابط، دون تكلفة إضافية عليك."
9. أضف رابط العمولة كما هو دون تغييره:
{data.affiliate_url}
10. أعد النتيجة بصيغة JSON صحيحة فقط، دون Markdown.

استخدم هذا الشكل:
{{
  "variants": [
    {{"title": "عنوان الإعلان", "content": "نص الإعلان", "hashtags": "الهاشتاقات"}},
    {{"title": "عنوان الإعلان", "content": "نص الإعلان", "hashtags": "الهاشتاقات"}},
    {{"title": "عنوان الإعلان", "content": "نص الإعلان", "hashtags": "الهاشتاقات"}}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "أنت كاتب إعلانات سعودي دقيق. التزم بالعربية السليمة واللهجة السعودية الطبيعية."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        variants = result.get("variants", [])

        if len(variants) != 3:
            raise ValueError("عدد الإعلانات غير صحيح")

        for ad in variants:
            ad["affiliate_url"] = data.affiliate_url
            ad["disclosure"] = (
                "إعلان بعمولة: قد أحصل على عمولة عند الشراء عبر هذا الرابط، "
                "دون تكلفة إضافية عليك."
            )

        return {
            "success": True,
            "country": "السعودية",
            "dialect": "السعودية",
            "disclosure": "إعلان بعمولة",
            "affiliate_url": data.affiliate_url,
            "variants": variants
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="تعذر إنشاء الإعلانات حالياً. حاول مرة أخرى."
        )
         
