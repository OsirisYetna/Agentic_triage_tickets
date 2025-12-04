# main.py
import json
import csv
from agent.llm import HFLLM
from agent.core import Agent

def main():
    # Loading the samples from the json file
    with open("data/sample_tickets.json", "r") as f:
        tickets = json.load(f)

    # Initialization of the llm and creation of the agent
    llm = HFLLM(model_name="google/flan-t5-small", max_tokens=128) # for the cpu, google/flan-t5-large for a GPU
    agent = Agent(llm)

    # storage for the event I will download in the csv
    results_list = []

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

        # Saving the results in a csv to upload it after
        results_list.append({
            "ticket_id": result['ticket_id'],
            "customer_tier": ticket.get('customer_tier', 'standard'),
            "created_at": ticket.get('created_at', ''),
            "channel": ticket.get('channel', 'email'),
            "summary": result['summary'],
            "urgency": result['urgency'],
            "urgency_conf": result.get('urgency_conf'),
            "action": result['action']
        })

    # Writing in a-the csv
    csv_file = "results_prompt_improved.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)

    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()

