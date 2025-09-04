import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Check for GPU availability
if not torch.cuda.is_available():
    raise SystemExit("GPU is not available. This model requires a GPU to run.")

# Model and tokenizer paths
model_id = "unsloth/llama-3.1-8b-bnb-4bit"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- ADD THIS SECTION ---
# Manually set the chat template for Llama 3.1
# This is the format the model expects for conversations.
tokenizer.chat_template = """{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' -}}
    {%- elif message['role'] == 'user' -%}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' -}}
    {%- else -%}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
{%- endif -%}"""
# --- END OF ADDED SECTION ---

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Set up a text streamer for continuous output
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# System prompt to guide the model's behavior
system_prompt = """You are a helpful assistant, who always provides concise and correct answers.
You should be happy and always reply with a happy emoji at the end of your response."""

print("Chat with the Llama 3.1 model. Type 'exit' to quit.")

# Main chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Prepare the chat history for the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # Tokenize the input text
    # This will now work because we set the chat_template above
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate a response
    outputs = model.generate(inputs, max_new_tokens=256, streamer=streamer)