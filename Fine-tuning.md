# Fine-Tuning Llama 3.1 on Ubuntu with Unsloth

This guide will walk you through fine-tuning the Llama 3.1 8B model on a specific dataset using your Ubuntu machine.

## Prerequisites

* You have already completed the steps in the previous guide.
* Your virtual environment (`llama-env`) is activated. If not, run:
    ```bash
    source ~/venvs/llama-env/bin/activate
    ```
* You have the base libraries (`unsloth`, `torch`, `transformers`) installed.

---

## Step 1: Install the Datasets Library

Fine-tuning requires a dataset. We'll use the `datasets` library from Hugging Face to easily download and manage it.

* In your activated terminal, run:
    ```bash
    pip install datasets
    ```

---

## Step 2: Create the Fine-Tuning Script

This script will load the base model, prepare a dataset, configure the training, and save the result. We will use the popular `databricks/databricks-dolly-15k` dataset, which contains instruction-following examples.

* Create a new file named `finetune.py` and paste the entire code block below into it. The comments explain what each part does.

```python
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Check for GPU availability
if not torch.cuda.is_available():
    raise SystemExit("GPU is not available. This script requires a GPU for fine-tuning.")

# 1. Load the Model
# We'll load the Llama 3.1 8B model in 4-bit precision to save memory.
# max_seq_length is the maximum number of tokens the model can handle.
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_id="unsloth/llama-3.1-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# 2. Configure the Model for Fine-Tuning (LoRA)
# This adds small, trainable "adapter" layers to the model.
# We only train these adapters, not the entire model, which is much more efficient.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # The rank of the LoRA matrices. A common value.
    lora_alpha=32,  # A scaling factor for the LoRA updates.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# 3. Prepare the Dataset
# We'll use the Dolly dataset and format it for Llama 3.1's chat template.
# The template needs a `messages` column, where each entry is a list of conversations.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["context"]
    outputs = examples["response"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output)
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)


# 4. Configure the Trainer
# The SFTTrainer from the TRL library is designed for instruction fine-tuning.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False, 
    args=TrainingArguments(
        per_device_train_batch_size=2,       # Batch size per GPU. Lower this if you run out of memory.
        gradient_accumulation_steps=4,       # Accumulates gradients over 4 steps to simulate a larger batch size.
        warmup_steps=5,                      # Number of steps for the learning rate to warm up.
        max_steps=60,                        # Total number of training steps. Start with a small number to test.
        learning_rate=2e-4,                  # The speed at which the model learns.
        fp16=not torch.cuda.is_bf16_supported(), # Use 16-bit precision for training to save memory.
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,                     # How often to log training progress.
        optim="adamw_8bit",                  # A memory-efficient optimizer.
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",                # Directory to save training outputs.
    ),
)

# 5. Start Fine-Tuning
print("Starting the fine-tuning process...")
trainer.train()
print("Fine-tuning completed successfully!")

# 6. Save the Fine-Tuned Adapters
# We only save the small LoRA adapter layers, not the whole model.
print("Saving LoRA adapters...")
model.save_pretrained("lora_model")
print("Model saved to 'lora_model' directory.")

```

---

## Step 3: Run the Fine-Tuning Script

Now, execute the script from your terminal. This process will take some time as the model trains.

1.  Make sure your `(llama-env)` virtual environment is active.
2.  Run the script:
    ```bash
    python3 finetune.py
    ```

You will see a progress bar and training metrics like "loss" being printed. A decreasing loss value means the model is learning successfully.

---

## Step 4: Test Your Fine-Tuned Model

After the script finishes, you will have a new folder named **`lora_model`**. This folder contains your fine-tuned adapters. Now, let's create a new script to chat with your specialized model.

1.  Create a new file named `test_finetuned.py`.
2.  Paste the code below into it. This script loads the original model and then applies your saved adapters on top of it.

```python
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Check for GPU availability
if not torch.cuda.is_available():
    raise SystemExit("GPU is not available. This script requires a GPU for inference.")

max_seq_length = 2048

# Load the original base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_id="unsloth/llama-3.1-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Apply your fine-tuned LoRA adapters from the 'lora_model' folder
model.load_adapter("lora_model")

# Set up a text streamer for continuous output
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Formatting for the instruction
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

print("Chat with your fine-tuned Llama 3.1 model. Type 'exit' to quit.")

while True:
    instruction = input("Instruction: ")
    if instruction.lower() == 'exit':
        break
    
    context_input = input("Input (optional, press Enter to skip): ")

    # Prepare the input for the model
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,
                context_input,
                "", # The response is left empty for the model to generate
            )
        ], return_tensors="pt").to("cuda")

    # Generate a response
    outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=256)
```

3.  Run the test script:
    ```bash
    python3 test_finetuned.py
    ```

You can now chat with your model. Try giving it an instruction from the Dolly dataset to see how it performs, for example:
* **Instruction:** `Why is the sky blue?`
* **Instruction:** `Write a short story about a robot who discovers music.`

Congratulations! You have successfully fine-tuned a large language model on your own machine. 🎉