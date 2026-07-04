import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not token:
    print("Please set HUGGINGFACEHUB_API_TOKEN in .env")
    exit(1)

headers = {"Authorization": f"Bearer {token}"}
print("Checking supported models on your HF account...")

models_to_test = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-2-9b-it",
    "google/gemma-1.1-7b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]

for model in models_to_test:
    url = f"https://router.huggingface.co/hf-inference/models/{model}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 10
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✅ {model} is fully supported!")
        else:
            print(f"❌ {model} failed: {response.json().get('error', response.text)}")
    except Exception as e:
        print(f"❌ {model} request failed: {e}")

