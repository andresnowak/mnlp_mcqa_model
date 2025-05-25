from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize model and tokenizer
model_name = "andresnowak/Qwen3-0.6B-instruction-finetuned"
model_name = "Qwen/Qwen3-0.6B-Base"  # or your finetuned version
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(prompt):
    # Tokenize the raw prompt directly
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the new tokens (excluding input)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

# Example usage for Terraform generation
# terraform_prompt = """Create a snippet of Terraform HCL code that create an AWS autoscaling group, and an ALB in front to expose an application to internet.

# Here's the Terraform code:"""
# response = generate_response(terraform_prompt)
# print(response)

# Interactive version
print("\nInteractive mode (type 'quit' to exit):")
while True:
    user_input = input("\nPrompt: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    print("\nResponse:", generate_response(user_input))