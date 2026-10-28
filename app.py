from fastapi import FastAPI
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

@app.post("/api/v1/generate-ads")
def generate_ads():
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for affiliate marketing."},
                {"role": "user", "content": "Generate a catchy affiliate marketing ad."}
            ]
        )
        return {"success": True, "ad": response.choices[0].message.content}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
def read_root():
    return {"message": "AI Affiliate Marketing API is running successfully with Groq!"}
    
@app.get("/models")
def list_models():
    try:
        models = client.models.list()
        return {"models": [model.id for model in models.data]}
    except Exception as e:
        return {"error": str(e)}
    
