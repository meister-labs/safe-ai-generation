from io import BytesIO
from hf import generate_response
import requests
import streamlit as st
from huggingface_hub import InferenceClient

import config

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
FILTER_API_URL = "https://filters-zeta.vercel.app/api/filter"

ENHANCE_SYS = (
    "Improve prompt for text-to-image generation. RETURN ONLY ENHANCED PROMPT. Do not include any other text."
    "Add subject, style, lighting, camera angle, and other details while keeping it safe."
)

NEGATIVE = "low quality, blurry, bad anatomy, disfigured, deformed, mutated, ugly, poorly drawn, bad proportions, extra limbs, cloned face, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, etc."

image_client = InferenceClient(provider="hf-inference", model=MODEL_ID, api_key=config.HF_API_KEY)

def check_prompt(prompt: str):
    try:
        response = requests.post(FILTER_API_URL, json={"prompt": prompt}, timeout=10)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            return {"ok": False, "reason": "Invalid response format from filter API."}
        return result
    except Exception as e:
        return {"ok": False, "reason": str(e)}

def enhance_prompt(prompt: str) -> str:
    output = generate_response(f"{ENHANCE_SYS}\n\nPrompt: {prompt}", temperature=0.4, max_tokens=220)

    return (output or prompt).strip()

def generate_image(prompt: str):
    filter_result = check_prompt(prompt)
    if not filter_result.get("ok"):
        return None, f"Prompt rejected by filter: {filter_result.get('reason', 'Unknown reason')}"

    try:
        return image_client.text_to_image(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            model = MODEL_ID,
        ), None
    except Exception as e:
        msg = str(e)

        if "negative prompt" in msg or "unexpected keyword" in msg:
            try:
                return image_client.text_to_image(
                    prompt=prompt,
                    model = MODEL_ID,
                ), None
            except Exception as e2:
                return str(e2)  