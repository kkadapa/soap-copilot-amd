# 🏥 SOAP Copilot — AMD MI300X

AI-powered clinical documentation running Llama 3.3 70B on AMD MI300X (192GB HBM3).

## What it does
Paste a doctor-patient conversation → get:
- Structured SOAP note
- ICD-10 diagnosis codes with confidence scores  
- Plain-language patient summary

## Architecture
Doctor-patient conversation ↓ SOAP Agent → Llama 3.3 70B (system prompt: medical scribe) ICD Agent → Llama 3.3 70B (system prompt: medical coder) Summary Agent → Llama 3.3 70B (system prompt: patient communication) ↓ Gradio UI → Hugging Face Space

## Hardware
- AMD MI300X — 192GB HBM3
- Single GPU serving 70B parameter model
- ROCm 7.2 + vLLM

## Run locally
```bash
pip install gradio openai
# Set your vLLM endpoint in app.py
python app.py
```

## Built for
AMD Developer Hackathon 2026
