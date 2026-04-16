import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from two_layer import HolographicExpansionMesh

# --- Configuration ---
# You'll need the HuggingFace repo ID or a local path to the PyTorch/Safetensors version of the 1-bit model.
# (If you only have the GGUF, you'd use a library like 'ctransformers' or convert it, 
# but the PyTorch hooking logic remains the exact same).
MODEL_ID = "PrismML/bonsai-8b" 
DIM = 4096  # Standard for 8B models, adjust if Bonsai differs
HIDDEN_DIM = DIM * 4

print("--- Initializing the Holographic Associative Memory (HAM) ---")
# Load the untrained/pre-trained HAM
ham_core = HolographicExpansionMesh(DIM, HIDDEN_DIM).cuda()
ham_core.eval() # Set to inference mode

print(f"--- Loading LLM ({MODEL_ID}) into VRAM ---")
# Load the tokenizer and the model directly onto the 4090
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="cuda", 
    torch_dtype=torch.float16 # We use float16 for the wrapper, but the internal weights are 1-bit
)

# --- The Neural Hook (The Brain Surgery) ---
def ham_intuition_hook(module, input, output):
    """
    This function intercepts the hidden states right before the LLM speaks.
    'output' is the continuous vector from the LLM's final transformer block.
    """
    # output[0] contains the hidden states of shape (batch_size, sequence_length, hidden_dim)
    llm_thought = output[0]
    
    # We only want to hook the VERY LAST token (the one predicting the next word)
    current_token_thought = llm_thought[:, -1, :] 
    
    # 1. Snap the LLM's thought to our 1-bit logic
    quantized_thought = torch.sign(current_token_thought)
    quantized_thought[quantized_thought == 0] = 1.0
    
    # 2. Route it through the HAM
    # The HAM synthesizes the next logical state based on its wave interference
    _, ham_intuition = ham_core(quantized_thought)
    
    # 3. Inject the HAM's thought back into the LLM's continuous stream
    # We cast it back to float16 so the LLM's unembedding head doesn't crash
    llm_thought[:, -1, :] = ham_intuition.to(torch.float16)
    
    # Return the mutated tensor back to the LLM
    return (llm_thought,) + output[1:]

# --- Registering the Hook ---
# We attach the hook to the final LayerNorm right before the lm_head
# (The exact attribute name might vary slightly depending on the specific model architecture, 
# e.g., model.model.norm or model.transformer.ln_f)
hook_handle = model.model.norm.register_forward_hook(ham_intuition_hook)
print("--- HAM Successfully Grafted to LLM Nervous System ---")

# --- The Interactive Loop ---
print("\nThe HAM is now in control. The LLM is just the translator.")
print("Type a prompt to test the intuition.")

while True:
    prompt = input("\nPrompt> ").strip()
    if prompt.lower() == 'exit':
        break
        
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate the response
    # The hook will automatically trigger on every single token generated
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Decode the final text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Strip the original prompt from the output for clean viewing
    clean_response = response[len(prompt):].strip()
    print(f"\nHAM Intuition: {clean_response}")

# Clean up the hook if you ever exit the script programmatically
hook_handle.remove()