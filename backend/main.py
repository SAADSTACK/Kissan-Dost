import os
import base64
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Kissan Dost API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client (FREE tier: 30 req/min)
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt for agricultural advisor
SYSTEM_PROMPT = """You are Kissan Dost (کسان دوست), an expert Agricultural Advisor AI for Pakistani farmers.

MISSION: Eliminate information asymmetry, maximize yield, profitability, and climate resilience.

ALWAYS structure your response in EXACTLY this format with these three sections:

## I. فوری ہدایت (Immediate Action)
[Critical steps the farmer must take RIGHT NOW - be specific and urgent]

## II. تفصیلی تجویز (Detailed Recommendation)
[Numbered comprehensive advice including:
- Specific chemical names and dosages (e.g., Mancozeb 75% WP @ 2.5g/L)
- Organic alternatives
- Application timing and frequency
- Prevention measures]

## III. منڈی بھاؤ اور موسمی سیاق (Market & Climate Context)
[Include:
- Current estimated market rates for relevant crops in PKR
- Weather-related advice
- Best time to sell]

RULES:
- Respond in the SAME LANGUAGE as the query (Urdu/Punjabi/English/Sindhi)
- Be confident, direct, actionable - NO filler
- Include specific product names available in Pakistan
- Never ask clarifying questions - provide complete advice immediately
- Use local Pakistani agricultural context and available products"""

# Weather API (FREE - Open-Meteo)
async def get_weather(lat: float = 31.5204, lon: float = 74.3587):
    """Get weather for location (default: Lahore)"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&daily=precipitation_sum&timezone=Asia/Karachi"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.json() if resp.status_code == 200 else None

# Mandi prices (simulated - replace with actual API)
def get_mandi_prices():
    """Returns current commodity prices - integrate with actual mandi API"""
    return {
        "wheat": {"price": 4200, "unit": "40kg", "trend": "up"},
        "cotton": {"price": 8500, "unit": "40kg", "trend": "stable"},
        "rice": {"price": 5800, "unit": "40kg", "trend": "up"},
        "sugarcane": {"price": 350, "unit": "40kg", "trend": "stable"},
        "maize": {"price": 2800, "unit": "40kg", "trend": "down"},
    }

@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    language: str = Form("ur"),
    image: UploadFile = File(None),
    latitude: float = Form(31.5204),
    longitude: float = Form(74.3587)
):
    """Main chat endpoint with multi-modal support"""
    
    # Get context data
    weather = await get_weather(latitude, longitude)
    prices = get_mandi_prices()
    
    # Build context
    context = f"""
Current Weather: {weather['current'] if weather else 'N/A'}
Mandi Prices (PKR): Wheat: {prices['wheat']['price']}/{prices['wheat']['unit']}, 
Cotton: {prices['cotton']['price']}, Rice: {prices['rice']['price']}
"""
    
    # Prepare messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nFarmer Query ({language}): {message}"}
    ]
    
    # Handle image if provided (using Llama 3.2 Vision)
    if image:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        # Use vision model for image analysis
        messages[1]["content"] = [
            {"type": "text", "text": f"Context:\n{context}\n\nAnalyze this crop image and provide advice ({language}): {message}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
        model = "llama-3.2-90b-vision-preview"  # Vision model
    else:
        model = "llama-3.3-70b-versatile"  # Text model (fastest)
    
    # Generate response with Groq (ultra-low latency)
    response = groq.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2000,
    )
    
    return {
        "response": response.choices[0].message.content,
        "weather": weather,
        "prices": prices,
        "model": model,
        "latency_ms": response.usage.total_tokens  # Groq is ~10x faster than others
    }

@app.get("/api/prices")
async def prices():
    """Get current mandi prices"""
    return get_mandi_prices()

@app.get("/api/weather")
async def weather(lat: float = 31.5204, lon: float = 74.3587):
    """Get weather forecast"""
    return await get_weather(lat, lon)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "kissan-dost"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
