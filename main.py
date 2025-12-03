# main.py
import json
from agent.llm import HFLLM
from agent.core import Agent

def main():
    # Loading the samples from the json file
    with open("samples/data.json", "r") as f:
        tickets = json.load(f)

    # Initialization of the llm and creation of the agent
    llm = HFLLM(model_name="google/flan-t5-small", max_tokens=128)
    agent = Agent(llm)

    # Dealing with each tickets
    for ticket in tickets:
        result = agent.run(ticket)
        
        print("="*80)
        print(f"Ticket ID      : {result['ticket_id']}")
        print(f"Customer Tier  : {ticket.get('customer_tier', 'standard')}")
        print(f"Created At     : {ticket.get('created_at', '')}")
        print(f"Channel        : {ticket.get('channel', 'email')}")
        print("-"*80)
        print(f"Summary        : {result['summary']}")
        print(f"Urgency        : {result['urgency']} (confidence={result.get('urgency_conf')})")
        print(f"Recommended Action: {result['action']}")
        print("-"*80)
        print("Trace of decisions:")
        for step in result['trace']:
            print(f"  Step {step['step']}: Action={step['action']} | Thoughts={step['thoughts']}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
