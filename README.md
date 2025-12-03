# Agentic Support Ticket Triage

## Description

This project demonstrates a **simple AI agent** for triaging customer support tickets. The agent uses a **HuggingFace LLM (Flan-T5)** to:

- Summarize tickets
- Score ticket urgency
- Recommend actions based on urgency

The agent is designed with **agentic behavior**: it can reason, plan, and choose actions dynamically. You can run it either on **CPU (smaller model)** or **GPU/Colab (larger, more reliable model)**.

---

## File Structure

```
Agentic_triage_tickets/
│
├─ agent/
│  ├─ __init__.py
│  ├─ llm.py              # Wrapper for HuggingFace LLM
│  ├─ core.py             # Main agent class that handles ticket triage
│  ├─ utils.py            # Helper functions: summarize_ticket, score_urgency, escalate_action
│  ├─ prompt.py           # CPU prompts
│  ├─ prompt_colab.py     # GPU/Colab prompts
│
├─ data/
│  └─ sample_tickets.json # Example tickets for testing
│
├─ main.py                # Main script for CPU testing
├─ main_colab.py          # Main script for GPU/Colab testing
├─ requirements.txt       # Python dependencies
└─ README.md              # This file
```

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd Agentic_triage_tickets
```

2. Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
transformers
torch
pandas
```

---

## Running the agent

### CPU (small model)

```bash
python3 main.py
```

- Uses `google/flan-t5-small`
- Suitable for testing on a local CPU
- May only reliably produce summaries due to model size limitations

### GPU / Colab (recommended)

```bash
python3 main_colab.py
```

- Uses `google/flan-t5-base` or `flan-t5-large` for better JSON outputs
- Supports all agent actions: summarize, score urgency, escalate action
- Faster and more reliable on Colab GPU

---

## Output

For each ticket, the agent outputs:

- **Summary**: concise description of the ticket
- **Urgency**: ticket priority (P0-P3) with confidence score
- **Recommended Action**: what the support team should do
- **Trace**: step-by-step reasoning of the agent

Example:

```
================================================================================
Ticket ID      : TICKET-001
Customer Tier  : VIP
Created At     : 2025-12-03T10:00:00
Channel        : email
--------------------------------------------------------------------------------
Summary        : The service is down and we are experiencing data loss.
Urgency        : P0 (confidence=0.95)
Recommended Action: Immediate: page on-call + create incident ticket; assign to the lead IT engineer notify account manager.
--------------------------------------------------------------------------------
Trace of decisions:
  Step 1: Action=summarize_ticket | Thoughts="The service is down..."
  Step 2: Action=score_urgency | Thoughts="Detected critical outage"
  Step 3: Action=escalate_action | Thoughts="Escalating to on-call team"
================================================================================
```

---

## Notes / Tips

- CPU testing is limited by the small LLM (flan-t5-small), so often only summaries are generated.
- For reliable action selection and JSON output, use GPU/Colab with flan-t5-base or large.
- You can export results to CSV for further analysis or reporting.