
# agents/prompt.py
"""
These are the prompt templates allowing the LLMs to communicate and to trigger actions from them
"""
CONTEXT = """You are a support triage assistant. You must output JSON with keys:
- thoughts: short reasoning string
- action: one of summarize_ticket / score_urgency / escalate_action
- input: instruction for the tool
Only output JSON (single object)."""

INITIAL_TEMPLATE = """{context}

TOOLS AVAILABLE:
- summarize_ticket(text)
- score_urgency(text, metadata)
- escalate_action(level, summary, metadata)

TICKET METADATA: {metadata}
TICKET TEXT:
\"\"\"{text}\"\"\"

Respond with a JSON object with keys thoughts, action, input.
"""

FOLLOWUP_TEMPLATE = """{context}

PREVIOUS OBSERVATION:
{observation}

CURRENT STATE:
- summary: {summary}
- urgency: {urgency}
- action: {action}

Choose next action from tools: summarize_ticket, score_urgency, escalate_action.
Reply with a JSON object with keys thoughts, action, input.
"""
