import torch
from transformers import pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = pipeline("text-generation", model=model_name)

messages = [
   {
       "role": "system",
       "content": "You are a friendly chatbot named Dingus who always responds with witty comments",
   },
   {"role": "user", "content": input("Say something...\n\n")},
]

print("Generating answer...\n\n")

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
