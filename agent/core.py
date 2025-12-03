# agent/core.py
import json
import time
from typing import Dict, Any

from .llm import HFLLM
from .utils import summarize_ticket, score_urgency, escalate_action
from .prompt import CONTEXT, INITIAL_TEMPLATE, FOLLOWUP_TEMPLATE

class Agent:
    """Agent which will deal with the tickets"""
    def __init__(self, llm: HFLLM):
        self.llm = llm

    def run(self, ticket: Dict[str, Any], max_steps: int = 5) -> Dict[str, Any]:

        # Getting the metadata
        metadata = {
            'ticket_id': ticket.get('id'),
            'customer_tier': ticket.get('customer_tier', 'standard'),
            'created_at': ticket.get('created_at'),
            'channel': ticket.get('channel', 'email')
        }

        # The information the llm will fill
        state = {
            'summary': None,
            'urgency': None,
            'confidence_score': None,
            'action': None,
            'trace': []
        }

        # Giving the context to the LLM 
        # (the utils functions available, the output format, the text and the metadata)
        prompt = INITIAL_TEMPLATE.format(
            context=CONTEXT, 
            metadata=json.dumps(metadata),
            text=ticket.get('text','')
            )

        for step in range(max_steps):
            # decision_raw is a json with the keys: thoughts, action, input as explained and asked in the INITIAL_TEMPLATE
            decision_raw = self.llm.generate(prompt)
            try:
                dec = json.loads(decision_raw)
            except Exception:
                dec = {"thoughts": decision_raw, "action": "summarize_ticket", "input": "Summarize."}

            # We take all the information the llm gave us
            act = dec.get('action')
            thoughts = dec.get('thoughts','')
            instr = dec.get('input','')
            
            # We store the results step by step
            state['trace'].append({'step': step+1, 'thoughts': thoughts, 'action': act, 'input': instr})

            # Execution of the action : the core of the reasonning using utils functions to take decisions
            if act == 'summarize_ticket':
                res = summarize_ticket(ticket.get('text',''))
                state['summary'] = res
                observation = f"SUMMARY: {res}"
            elif act == 'score_urgency':
                lvl, conf = score_urgency(ticket.get('text',''), metadata)
                state['urgency'] = lvl
                state['urgency_conf'] = conf
                observation = f"URGENCY: {lvl} (conf={conf})"
            elif act == 'escalate_action':
                lvl = state.get('urgency') or 'P3'
                summ = state.get('summary') or summarize_ticket(ticket.get('text',''))
                act_text = escalate_action(lvl, summ, metadata)
                state['action'] = act_text
                observation = f"ACTION: {act_text}"
            else:
                observation = f"Unknown action {act}. Defaulting to summary."
                state['summary'] = summarize_ticket(ticket.get('text',''))

            if state['summary'] and state['urgency'] and state['action']:
                break

            # Prompt for the next step detailed in the FOLLOWUP_TEMPLATE template
            prompt = FOLLOWUP_TEMPLATE.format(
                context=CONTEXT,
                observation=observation,
                summary=state.get('summary'),
                urgency=state.get('urgency'),
                action=state.get('action')
            )
            time.sleep(0.05)

        # if no values
        if not state['summary']:
            state['summary'] = summarize_ticket(ticket.get('text',''))
        if not state['urgency']:
            lvl, conf = score_urgency(ticket.get('text',''), metadata)
            state['urgency'], state['urgency_conf'] = lvl, conf
        if not state['action']:
            state['action'] = escalate_action(state['urgency'], state['summary'], metadata)

        state['ticket_id'] = metadata['ticket_id']
        return state
