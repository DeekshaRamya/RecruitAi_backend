import os
import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "")
model = "gemini-3.1-flash-tts-preview"

url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}

payload = {
    "contents": [
        {
            "parts": [{"text": "Hello, welcome to your interview. Can you please tell me about yourself?"}]
        }
    ],
    "generationConfig": {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": "Puck"
                }
            }
        }
    }
}

try:
    response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        print("Response parts keys:")
        for idx, part in enumerate(parts):
            print(f"Part {idx}: {part.keys()}")
            if "inlineData" in part:
                print(f"Found inlineData! mimeType: {part['inlineData']['mimeType']}, data length: {len(part['inlineData']['data'])}")
    else:
        print("Error response:", response.text)
except Exception as e:
    print("Exception occurred:", e)
