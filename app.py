import gradio as gr
import json
import time
from openai import OpenAI

client = OpenAI(base_url="https://opened-cube-length-corporation.trycloudflare.com/v1", api_key="not-needed")
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

DEMOS = {
    "Headache": "Doctor: What brings you in today?\nPatient: I've had a throbbing headache for three days, mostly in my forehead.\nDoctor: On a scale of 1 to 10 how bad is the pain?\nPatient: About a 7. Ibuprofen helps a little but it keeps coming back.\nDoctor: Any nausea or light sensitivity?\nPatient: Some light sensitivity yes. No nausea.\nDoctor: Any fever or stiff neck?\nPatient: No fever. Neck feels fine.\nDoctor: Any recent stress or poor sleep?\nPatient: Very stressful week at work and not sleeping well.",
    "Chest Pain": "Doctor: What brings you in?\nPatient: Chest pain when I climb stairs, started about 2 weeks ago.\nDoctor: Can you describe the pain?\nPatient: Pressure, like someone sitting on my chest. Sometimes goes to my left arm.\nDoctor: How long does it last?\nPatient: 5 to 10 minutes, goes away when I rest.\nDoctor: Do you smoke?\nPatient: Yes, a pack a day for 20 years.\nDoctor: Any family history of heart disease?\nPatient: My father had a heart attack at 58.",
    "Diabetes Follow-up": "Doctor: How have you been managing your blood sugar?\nPatient: Pretty well I think. I check it every morning.\nDoctor: What are your typical fasting readings?\nPatient: Usually between 130 and 160.\nDoctor: Are you taking your metformin as prescribed?\nPatient: Yes twice a day with meals.\nDoctor: Any episodes of low blood sugar?\nPatient: One last month, felt shaky, drank some juice and it passed.\nDoctor: How is your diet and exercise?\nPatient: Walking 30 minutes daily. Diet could be better.",
    "Hypertension": "Doctor: How have your blood pressure readings been at home?\nPatient: Higher than I would like, usually around 150 over 95.\nDoctor: Are you taking your lisinopril every day?\nPatient: Most days. Sometimes I forget.\nDoctor: Any headaches or dizziness?\nPatient: Occasional headaches in the morning.\nDoctor: How is your salt intake?\nPatient: I try to limit it but I eat out a lot.\nDoctor: Any ankle swelling?\nPatient: A little at the end of the day.",
}

def soap_agent(conversation):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are an expert medical scribe. Generate a structured SOAP note from the doctor-patient conversation.

Format exactly as:
SUBJECTIVE: [chief complaint, HPI, symptoms, PMH, medications, allergies]
OBJECTIVE: [estimated vitals based on context, physical exam findings]
ASSESSMENT: [primary diagnosis with clinical reasoning]
PLAN: [numbered treatment steps, medications, follow-up, patient education]"""},
            {"role": "user", "content": f"Generate a SOAP note from this conversation:\n\n{conversation}"}
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()

def icd_agent(soap_note):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are a certified medical coder. Extract ICD-10 diagnosis codes from SOAP notes.
Return ONLY a JSON array, no other text:
[{"code": "G44.209", "description": "Tension-type headache, unspecified", "confidence": 94}]
Include primary diagnosis first, then secondary codes. Confidence 0-100."""},
            {"role": "user", "content": f"Extract ICD-10 codes from:\n\n{soap_note}"}
        ],
        temperature=0.1,
        max_tokens=400,
    )
    text = resp.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        codes = json.loads(text.strip())
        lines = []
        for c in codes:
            bar = "█" * (c['confidence'] // 10) + "░" * (10 - c['confidence'] // 10)
        lines.append(f"• {c['code']}  {c['description']}\n  [{bar}] {c['confidence']}% confidence")
        return "\n\n".join(lines)
    except:
        return text

def summary_agent(soap_note):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a patient communication specialist. Write a clear, warm, 3-4 sentence visit summary in plain language a patient can understand. No medical jargon. Start with what was discussed, then what was decided, then next steps."},
            {"role": "user", "content": f"Write a patient-friendly summary of this visit:\n\n{soap_note}"}
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

def run_pipeline(demo_choice, conversation):
    if demo_choice != "Custom input":
        conversation = DEMOS[demo_choice]
    if not conversation.strip():
        return "⚠️ Please enter a conversation or select a demo case.", "", "", ""
    t0 = time.time()
    try:
        soap = soap_agent(conversation)
        icd = icd_agent(soap)
        summary = summary_agent(soap)
        elapsed = round(time.time() - t0, 1)
        status = f"✅ Generated in {elapsed}s · AMD MI300X · Llama 3.3 70B"
        return soap, icd, summary, status
    except Exception as e:
        return f"Error: {e}", "", "", "❌ Failed"

with gr.Blocks(
    title="SOAP Copilot — AMD MI300X",
    theme=gr.themes.Soft(primary_hue="purple"),
    css="footer{display:none}"
) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:1.5rem 0 0.5rem">
      <h1 style="font-size:2rem;margin:0">🏥 SOAP Copilot</h1>
      <p style="color:#888;margin:0.5rem 0">AI-powered clinical documentation · AMD MI300X · Llama 3.3 70B</p>
      <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:10px">
        <span style="background:#f0e8ff;color:#533AB7;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:500">⚡ AMD MI300X 192GB HBM3</span>
        <span style="background:#e8f5e9;color:#2e7d32;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:500">🦙 Llama 3.3 70B Instruct</span>
        <span style="background:#fff3e0;color:#e65100;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:500">🏥 3-Agent Medical Pipeline</span>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            demo_choice = gr.Radio(
                choices=["Custom input"] + list(DEMOS.keys()),
                value="Headache",
                label="Quick demo cases",
            )
            conversation = gr.Textbox(
                label="Doctor-patient conversation",
                placeholder="Doctor: What brings you in today?\nPatient: ...",
                lines=12,
                value=DEMOS["Headache"],
            )
            run_btn = gr.Button(
                "🚀 Generate Clinical Documentation",
                variant="primary",
                size="lg"
            )
            status = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Column(scale=2):
            with gr.Tab("📋 SOAP Note"):
                soap_out = gr.Textbox(
                    label="Structured SOAP Note",
                    lines=16,
                    interactive=False
                )
            with gr.Tab("🏷️ ICD-10 Codes"):
                icd_out = gr.Textbox(
                    label="Diagnosis Codes + Confidence",
                    lines=10,
                    interactive=False
                )
            with gr.Tab("📝 Patient Summary"):
                sum_out = gr.Textbox(
                    label="Plain-language Patient Summary",
                    lines=6,
                    interactive=False
                )

    def update_conversation(demo_choice):
        if demo_choice != "Custom input":
            return DEMOS[demo_choice]
        return ""

    demo_choice.change(fn=update_conversation, inputs=demo_choice, outputs=conversation)

    run_btn.click(
        fn=run_pipeline,
        inputs=[demo_choice, conversation],
        outputs=[soap_out, icd_out, sum_out, status],
    )

    gr.HTML("""
    <div style="text-align:center;margin-top:1.5rem;padding:1rem;border-top:1px solid #eee;color:#999;font-size:12px">
      ⚠️ For educational purposes only. Not medical advice.<br>
      Built on AMD Developer Cloud · Open source · 
      <a href="https://huggingface.co/lablab-ai-amd-developer-hackathon" target="_blank">Hugging Face Organization</a>
    </div>
    """)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
