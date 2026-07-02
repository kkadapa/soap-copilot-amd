# 🏥 SOAP Copilot — AMD MI300X

> AI-powered clinical documentation using Llama 3.3 70B on AMD MI300X (192GB HBM3). Paste a doctor-patient conversation, get a structured SOAP note, ICD-10 codes, and a plain-language patient summary in seconds.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/soap-copilot-amd)
[![AMD MI300X](https://img.shields.io/badge/AMD-MI300X%20192GB-red)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
[![Llama 3.3 70B](https://img.shields.io/badge/Meta-Llama%203.3%2070B-blue)](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-orange)](https://rocm.docs.amd.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Architecture](#architecture)
- [Hardware](#hardware)
- [Agent Pipeline](#agent-pipeline)
- [Data Flow](#data-flow)
- [Deployment Architecture](#deployment-architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Built With](#built-with)

---

## 🚨 Problem Statement

Physicians spend **2 hours per day** on clinical documentation — time taken directly from patient care. Manual SOAP note writing is:

- Repetitive and mentally exhausting
- Error-prone under time pressure
- A leading cause of physician burnout
- Inconsistent across providers

**The cost:** $8.3B annually in lost physician productivity in the US alone.

---

## 💡 Solution

SOAP Copilot is a **multi-agent AI system** that transforms raw doctor-patient conversations into structured clinical documentation in seconds.

```
Input:  Raw conversation transcript
Output: SOAP Note + ICD-10 Codes + Patient Summary
Time:   ~15-30 seconds on AMD MI300X
```

The system runs entirely on **open-source models** on **AMD hardware** — no proprietary cloud APIs, no PHI leaving your infrastructure.

---

## 🏗️ Architecture

### System Overview

```mermaid
graph TB
    subgraph INPUT["📥 Input Layer"]
        A[Doctor-Patient Conversation]
    end

    subgraph UI["🖥️ User Interface"]
        B[Gradio Web App<br/>Hugging Face Space]
    end

    subgraph TUNNEL["🔒 Secure Tunnel"]
        C[Cloudflare Tunnel<br/>HTTPS → localhost:8000]
    end

    subgraph GPU["⚡ AMD MI300X — 192GB HBM3"]
        D[vLLM Inference Server<br/>v0.11.2 · ROCm 7.2]
        E[Llama 3.3 70B Instruct<br/>BFloat16 · 8192 ctx]
        D --> E
    end

    subgraph AGENTS["🤖 Agent Pipeline"]
        F[SOAP Generator Agent<br/>Medical Scribe Role]
        G[ICD Coding Agent<br/>Medical Coder Role]
        H[Summary Agent<br/>Patient Communication Role]
    end

    subgraph OUTPUT["📤 Output Layer"]
        I[Structured SOAP Note]
        J[ICD-10 Codes + Confidence]
        K[Plain-language Summary]
    end

    A --> B
    B --> C
    C --> D
    E --> F
    F --> G
    F --> H
    F --> I
    G --> J
    H --> K
```

### Technology Stack

```mermaid
graph LR
    subgraph HARDWARE["Hardware"]
        H1[AMD MI300X<br/>192GB HBM3<br/>750W TDP]
    end

    subgraph RUNTIME["Runtime"]
        R1[ROCm 7.2]
        R2[Docker Container]
        R3[vLLM 0.11.2]
        R1 --> R2 --> R3
    end

    subgraph MODEL["Model"]
        M1[Llama 3.3 70B Instruct]
        M2[BFloat16 Precision]
        M3[8192 Token Context]
        M1 --> M2 --> M3
    end

    subgraph APP["Application"]
        A1[Python 3.12]
        A2[Gradio 6.x]
        A3[OpenAI SDK]
        A1 --> A2
        A1 --> A3
    end

    HARDWARE --> RUNTIME --> MODEL --> APP
```

---

## ⚡ Hardware

The AMD MI300X is the key differentiator that makes this system possible at this quality level.

```mermaid
graph TD
    subgraph MI300X["AMD MI300X — Single Chip"]
        subgraph HBM["HBM3 Memory — 192GB"]
            M1[Model Weights<br/>~140GB in BF16]
            M2[KV Cache<br/>~30GB]
            M3[Activations<br/>~22GB]
        end
        subgraph COMPUTE["Compute Units"]
            C1[304 CUs<br/>5.3 TFLOPS BF16]
        end
        subgraph BANDWIDTH["Memory Bandwidth"]
            B1[5.3 TB/s<br/>HBM3]
        end
    end

    subgraph COMPARISON["Why MI300X Matters"]
        D1["Consumer GPU: 24GB VRAM<br/>❌ Cannot run 70B"]
        D2["Cloud A100: 80GB VRAM<br/>❌ Cannot run 70B in BF16"]
        D3["AMD MI300X: 192GB VRAM<br/>✅ Runs 70B with headroom"]
    end
```

| Spec | Value |
|------|-------|
| Architecture | CDNA3 |
| VRAM | 192 GB HBM3 |
| Memory Bandwidth | 5.3 TB/s |
| TDP | 750W |
| ROCm Version | 7.2.3 |
| vLLM Version | 0.11.2 |
| Model Precision | BFloat16 |
| GPU Utilization at load | 100% |

---

## 🤖 Agent Pipeline

Three specialized agents, each with a different system prompt, call the same 70B model with different roles. Sequential pipeline ensures each agent builds on the previous output.

```mermaid
sequenceDiagram
    actor User
    participant UI as Gradio UI
    participant CF as Cloudflare Tunnel
    participant vLLM as vLLM Server<br/>(AMD MI300X)
    participant LLM as Llama 3.3 70B

    User->>UI: Paste conversation + click Generate
    UI->>CF: POST /v1/chat/completions
    CF->>vLLM: Forward request (HTTPS→HTTP)

    Note over vLLM,LLM: Agent 1 — SOAP Generator
    vLLM->>LLM: system: "You are an expert medical scribe..."<br/>user: [conversation]
    LLM-->>vLLM: SUBJECTIVE / OBJECTIVE / ASSESSMENT / PLAN
    vLLM-->>UI: SOAP Note

    Note over vLLM,LLM: Agent 2 — ICD Coder
    vLLM->>LLM: system: "You are a certified medical coder..."<br/>user: [SOAP note]
    LLM-->>vLLM: JSON array of ICD-10 codes + confidence
    vLLM-->>UI: Structured ICD codes

    Note over vLLM,LLM: Agent 3 — Summary Writer
    vLLM->>LLM: system: "You are a patient communication specialist..."<br/>user: [SOAP note]
    LLM-->>vLLM: Plain-language summary
    vLLM-->>UI: Patient summary

    UI-->>User: SOAP + ICD + Summary displayed
```

### Agent Specifications

```mermaid
graph TB
    subgraph SOAP["🩺 Agent 1: SOAP Generator"]
        S1[Role: Expert Medical Scribe]
        S2[Input: Raw conversation]
        S3[Output: S/O/A/P structure]
        S4[Temperature: 0.2<br/>Max tokens: 800]
        S1 --> S2 --> S3
        S4 -.-> S3
    end

    subgraph ICD["🏷️ Agent 2: ICD Coder"]
        I1[Role: Certified Medical Coder]
        I2[Input: SOAP note]
        I3[Output: JSON ICD-10 array<br/>with confidence scores]
        I4[Temperature: 0.1<br/>Max tokens: 400]
        I1 --> I2 --> I3
        I4 -.-> I3
    end

    subgraph SUM["📝 Agent 3: Summary Writer"]
        U1[Role: Patient Communication Specialist]
        U2[Input: SOAP note]
        U3[Output: 3-4 sentence<br/>plain-language summary]
        U4[Temperature: 0.3<br/>Max tokens: 300]
        U1 --> U2 --> U3
        U4 -.-> U3
    end

    SOAP -->|SOAP note passed| ICD
    SOAP -->|SOAP note passed| SUM
```

---

## 🔄 Data Flow

```mermaid
flowchart TD
    A([User opens HF Space]) --> B[Selects demo case\nor pastes conversation]
    B --> C{Input valid?}
    C -->|No| D[Show warning message]
    C -->|Yes| E[Start timer]

    E --> F[Call SOAP Agent\nvia OpenAI SDK]
    F --> G[POST to Cloudflare URL\n/v1/chat/completions]
    G --> H[Cloudflare forwards to\nlocalhost:8000]
    H --> I[vLLM routes to\nLlama 3.3 70B]
    I --> J[GPU processes\n~10-15 seconds]
    J --> K[Return SOAP note text]

    K --> L[Call ICD Agent\nwith SOAP as input]
    L --> M[Return JSON array\nof ICD-10 codes]

    M --> N[Call Summary Agent\nwith SOAP as input]
    N --> O[Return plain-language\npatient summary]

    O --> P[Parse ICD JSON\nformat with confidence bars]
    P --> Q[Display all three outputs\nin tabbed interface]
    Q --> R[Show elapsed time\nand token count]
    R --> S([User sees results])

    style J fill:#ff6b35,color:#fff
    style I fill:#cc2936,color:#fff
```

---

## 🚀 Deployment Architecture

```mermaid
graph TB
    subgraph INTERNET["🌐 Internet"]
        HF[Hugging Face Space<br/>ZeroGPU · Gradio UI]
        USER[End User Browser]
        USER -->|HTTPS| HF
    end

    subgraph CLOUDFLARE["☁️ Cloudflare Network"]
        CF[Cloudflare Tunnel<br/>opened-cube-length-corporation<br/>.trycloudflare.com]
        HF -->|API calls via HTTPS| CF
    end

    subgraph SERVER["🖥️ AMD Developer Cloud"]
        subgraph HOST["Bare Metal Host"]
            CFD[cloudflared daemon<br/>localhost tunnel agent]
            CF -->|Encrypted tunnel| CFD

            subgraph DOCKER["Docker Containers"]
                subgraph VLLM["rocm/vllm:latest"]
                    VLLM_SRV[vLLM API Server<br/>port 8000]
                    CFD -->|localhost:8000| VLLM_SRV
                end
                subgraph QDRANT["qdrant/qdrant"]
                    QD[Qdrant Vector DB<br/>port 6333]
                end
                subgraph TRAIN["rocm/pytorch:latest"]
                    TR[Training Container<br/>nifty_satoshi]
                end
            end

            subgraph GPU["AMD MI300X"]
                VRAM[192GB HBM3<br/>Llama 3.3 70B loaded<br/>~140GB utilized]
                VLLM_SRV --> VRAM
            end
        end
    end
```

---

## 🛠️ Setup & Installation

### Prerequisites

- AMD MI300X GPU (or compatible ROCm GPU)
- Docker with ROCm support
- ROCm 6.0+
- Hugging Face account with Llama 3.3 70B access

### Step 1 — Verify GPU

```bash
rocm-smi
# Should show MI300X with 192GB VRAM
```

### Step 2 — Launch vLLM with 70B

```bash
docker run -d \
  --name vllm_fresh \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size 16G \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  rocm/vllm:latest \
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85

# Wait for startup (~3 min)
docker logs -f vllm_fresh | grep "startup complete"
```

### Step 3 — Expose via Cloudflare Tunnel

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb
cloudflared tunnel --url http://localhost:8000
# Copy the https://xxxx.trycloudflare.com URL
```

### Step 4 — Launch Training Container

```bash
docker run -it \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size 16G \
  -v $HOME/workspace:/workspace \
  -w /workspace \
  rocm/pytorch:latest bash
```

### Step 5 — Install Dependencies

```bash
pip install transformers trl peft accelerate datasets \
            sentencepiece gradio openai huggingface_hub --upgrade
```

### Step 6 — Configure and Run App

```bash
# Update endpoint in app.py
sed -i 's|http://localhost:8000/v1|https://YOUR-TUNNEL.trycloudflare.com/v1|g' app.py

# Run
python app.py
```

---

## 💻 Usage

### Quick Demo

1. Open the [Hugging Face Space](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/soap-copilot-amd)
2. Select a demo case (Headache, Chest Pain, Diabetes, Hypertension)
3. Click **Generate Clinical Documentation**
4. View results across three tabs: SOAP Note, ICD-10 Codes, Patient Summary

### Custom Input

Paste any doctor-patient conversation in this format:

```
Doctor: What brings you in today?
Patient: I've had a throbbing headache for three days...
Doctor: On a scale of 1-10, how bad is the pain?
Patient: About a 7. Ibuprofen helps a little...
```

### Example Output

**SOAP Note:**
```
SUBJECTIVE: 34yo female presents with 3-day h/o throbbing frontal 
headache, severity 7/10. Reports photophobia. No nausea or fever. 
Significant psychosocial stressor (work stress) and sleep disruption.

OBJECTIVE: VS: BP 118/76, HR 72, Temp 98.4°F. Alert and oriented x3. 
HEENT: No papilledema. Neck: Supple. Neuro: CN II-XII intact.

ASSESSMENT: Tension-type headache, episodic, likely exacerbated by 
psychosocial stress and sleep disruption.

PLAN:
1. Ibuprofen 600mg q6h PRN with food
2. Referral to behavioral health for stress management
3. Sleep hygiene education provided
4. Return if headache worsens or neurological symptoms develop
```

**ICD-10 Codes:**
```
• G44.209 — Tension-type headache, unspecified       [████████░░] 94%
• F43.10  — Post-traumatic stress, unspecified        [███████░░░] 71%
• G47.00  — Insomnia, unspecified                    [██████░░░░] 68%
```

**Patient Summary:**
```
Today we discussed the throbbing headache you've been experiencing 
for the past three days. We believe it's a tension headache brought 
on by work stress and disrupted sleep. We recommend taking ibuprofen 
as needed with food and have referred you to behavioral health for 
stress management support. Please return if your symptoms worsen or 
you develop any new neurological symptoms.
```

---

## 📊 Dataset

80 synthetic doctor-patient conversations generated using Llama 3.3 70B covering:

| Scenario | Count |
|----------|-------|
| Headache (various types) | 12 |
| Chest pain / cardiac | 8 |
| Diabetes management | 8 |
| Hypertension follow-up | 8 |
| Musculoskeletal pain | 8 |
| Respiratory complaints | 8 |
| Mental health | 8 |
| Dermatology | 8 |
| GI complaints | 8 |
| Preventive care | 4 |

### Dataset Format

```json
{
  "instruction": "Generate a SOAP note and ICD-10 code from this clinical conversation.",
  "input": "Doctor: What brings you in today?\nPatient: ...",
  "output": "SUBJECTIVE: ...\nOBJECTIVE: ...\nASSESSMENT: ...\nPLAN: ...\nICD-10: G44.209"
}
```

---

## 🎯 Training

LoRA fine-tuning configuration (for custom specialist model):

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    max_seq_length=2048,
    gradient_checkpointing=True,
)
```

### Training Hardware Utilization

```
GPU:   AMD MI300X
Power: 751W (at cap — fully loaded)
VRAM:  192GB HBM3
GPU%:  100% during training
Temp:  66°C (healthy operating range)
Time:  ~15 minutes for 80 examples × 3 epochs
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Average generation time | 15-30 seconds |
| SOAP note quality | Clinically structured |
| ICD-10 accuracy | Validated against WHO ICD-10 |
| Supported specialties | General medicine |
| Context window | 8,192 tokens |
| Concurrent requests | ~11x at full context |

---

## 🔧 Built With

| Component | Technology |
|-----------|------------|
| LLM | Meta Llama 3.3 70B Instruct |
| Inference Server | vLLM 0.11.2 |
| GPU | AMD MI300X 192GB |
| GPU Runtime | ROCm 7.2.3 |
| Training Framework | HuggingFace TRL + PEFT |
| UI Framework | Gradio 6.x |
| Deployment | Hugging Face Spaces |
| Tunnel | Cloudflare Tunnel |
| Vector DB | Qdrant (available for RAG extension) |
| Container | Docker + rocm/pytorch:latest |

---

## 🗺️ Roadmap

```mermaid
gantt
    title SOAP Copilot Roadmap
    dateFormat  YYYY-MM-DD
    section MVP (Done)
    3-agent pipeline         :done, 2026-05-10, 1d
    Gradio UI                :done, 2026-05-10, 1d
    HF Space deployment      :done, 2026-05-11, 1d
    section Next
    Fine-tuned 8B specialist :active, 2026-05-12, 7d
    Qdrant RAG integration   :2026-05-19, 7d
    Voice input support      :2026-05-26, 7d
    section Future
    EHR integration          :2026-06-02, 14d
    Multi-specialty support  :2026-06-16, 14d
    On-premise HIPAA deploy  :2026-07-01, 14d
```

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not intended for clinical use and should not be used to make medical decisions. Always consult a qualified healthcare professional.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🏆 Built for AMD Developer Hackathon 2026

Running Llama 3.3 70B on a single AMD MI300X — 192GB HBM3 — serving three specialized medical AI agents simultaneously. This workload is physically impossible on consumer hardware.

**The AMD MI300X is not just a hardware choice — it's what makes the system work.**

