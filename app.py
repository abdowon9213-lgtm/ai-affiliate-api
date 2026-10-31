import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="AI Affiliate Marketing API")

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class AdRequest(BaseModel):
    product_name: str = Field(..., min_length=1, description="Product Name")
    features: str = Field(..., min_length=1, description="Product Features")
    target_audience: str = Field(..., min_length=1, description="Target Audience")
    language: str = Field(..., description="Language: ar_ly / ar_msa / en")

LANG_INSTRUCTIONS = {
    "ar_msa": "Modern Standard Arabic (Professional and engaging marketing style).",
    "ar_ly": "Libyan Arabic dialect (Natural, engaging, and tailored for Libyan consumers).",
    "en": "English (Professional and persuasive marketing language)."
}

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return {"message": "API is running successfully! Please ensure index.html is in the same directory."}

@app.post("/api/v1/generate-ads")
def generate_ads(payload: AdRequest):
    try:
        lang_instruction = LANG_INSTRUCTIONS.get(payload.language, LANG_INSTRUCTIONS["ar_msa"])

        system_prompt = (
            "You are a professional digital marketing and ad copy expert.\n"
            "Strictly follow these rules:\n"
            "1. Prevent hallucinations: Use only the provided product information and features. Do not invent or add unmentioned details.\n"
            "2. Language compliance: Write the ad exclusively in the following language/dialect: " f"{lang_instruction}\n"
            "3. Focus on highlighting product benefits to the target audience in a direct and engaging way."
        )

        user_content = (
            f"Product Name: {payload.product_name}\n"
            f"Product Features: {payload.features}\n"
            f"Target Audience: {payload.target_audience}\n"
        )

        response = client.chat.completions.create(
            model="llama3-70b-8192"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.4,
            max_tokens=600
        )
        
        return {"success": True, "ad": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    try:
        models = client.models.list()
        return {"models": [model.id for model in models.data]}
    except Exception as e:
        return {"error": str(e)}
