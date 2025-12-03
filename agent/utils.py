# agent/utils.py
"""
Utils function for concision and readability that will summarize and trigger actions
"""
import re
from typing import Dict, Any, Tuple
from .llm import HFLLM

def summarize_ticket_with_llm(text: str) -> str:
    """Summarizing the tickets with an LLM
    Allowing more complex, natural and richer summarizing of each tickets
    But it take too much time and it is long to compute only with a CPU
    For the use case, I will not use it
    """
    llm = HFLLM()
    prompt = f"Summarize this ticket in one concise sentence:\n{text}"
    return llm.generate(prompt)

def summarize_ticket(text: str) -> str:
    """Simple summurize take the 25 first words"""

    # If the first sentence of the ticket is small enough, we return it
    sentences = re.split(r'(?<=[\.\?\!])\s+', text) # Spliting the text in list of sentences

    if sentences and len(sentences[0].split()) <= 25:
        return sentences[0].strip()
    
    # else, we "summarize it" with the 25 fires words
    words = text.split()
    return " ".join(words[:25]).strip() + "..."

def score_urgency(text: str, metadata: Dict[str, Any]) -> Tuple[str, float]:
    """Assigns an urgency level to a support ticket based on its text content and metadata.
    Output: 
    - level: str : urgency level of the ticket (P0, P1, P2, P3)
    - confidence: float : a float between 0 and 1 indicating how sure the scoring is
    Helps the agent prioritize tickets and optionally take extra steps if confidence is low
    """
    
    txt = text.lower()
    vip = metadata.get('customer_tier', '').lower() in ('vip', 'enterprise')

    if any(k in txt for k in ['outage','service is down','data loss','payment failed']):
        return 'P0', 0.95
    if any(k in txt for k in ['error','cannot','unable','failed']):
        return ('P0', 0.9) if vip else ('P1', 0.8)
    if any(k in txt for k in ['slow','latency','degraded']):
        return 'P2', 0.7
    return 'P3', 0.6

def escalate_action(level: str, summary: str, metadata: Dict[str, Any]) -> str:
    """Return a suggested action in function of the urgency level"""
    if level == 'P0':
        return "Immediate: page on-call + create incident ticket; assign to the lead IT engineer notify account manager."
    if level == 'P1':
        return "High: assign it to engineer of the team 2 and notify Product Manager and the account manager."
    if level == 'P2':
        return "Medium: request logs from customer and schedule triage."
    return "Low: auto-reply and request more info if needed."
