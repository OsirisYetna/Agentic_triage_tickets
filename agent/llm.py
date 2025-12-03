# agent/llm.py
""" 
This file defines a wrapper class HFLLM for HuggingFace Transformers (T5/FLAN).
It allows the agent to use LLMs with minimal code and clear abstraction.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# AutoTokenizer to transform the input text to token understandable by the llm
# AutoModelForSeq2SeqLM to generate text

class HFLLM:
    """Simple Wrapper for HuggingFace Transformers (T5/FLAN)"""
    # google/flan-t5-base is light and run well on CPU
    def __init__(self, model_name="google/flan-t5-small", max_tokens=128, device=None): 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") # to use Colab's GPUs
        self.model.to(self.device)

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            do_sample=False  
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
