# Running Llama 3.1 on Ubuntu with Unsloth

This guide provides a step-by-step walkthrough for setting up and running the Llama 3.1 8B model on an Ubuntu machine with an NVIDIA GPU.

## Prerequisites

Before you begin, ensure you have the following:
* An Ubuntu system.
* Python 3 installed (`python3`).
* An NVIDIA GPU (like your RTX 5060) with the appropriate drivers installed.

---

## Step 1: Create and Activate a Virtual Environment

A virtual environment keeps your Python project dependencies isolated. We will create a central one that you can reuse for other projects.

1.  **Create a central directory for your virtual environments:**
    Open a terminal and run:
    ```bash
    mkdir ~/venvs
    ```

2.  **Create a new virtual environment named `ai_test`:**
    ```bash
    python3 -m venv ~/venvs/ai_test
    ```

3.  **Activate the virtual environment:**
    To start using it, you must activate it in your terminal session.
    ```bash
    source ~/venvs/ai_test/bin/activate
    ```
    Your terminal prompt will now be prefixed with `(ai_test)`, indicating that the environment is active.

---

## Step 2: Install Required Libraries

With the environment active, install Unsloth, PyTorch, and other necessary libraries from Hugging Face.

```bash
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
```

---

## Step 3: Create the Python Script to Run the Model

Create a new file named `chat.py` 

```python

# Model and tokenizer paths
model_id = "unsloth/llama-3.1-8b-bnb-4bit"
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

```

---

## Step 4: Run the Chatbot

Now, run the script from your terminal.

```bash
python3 chat.py
```

The first time you run this, it will download the model, which may take several minutes depending on your internet connection. Subsequent runs will be much faster as the model will be loaded from the local cache.

Once the model is loaded, you will see the prompt:
`Chat with the Llama 3.1 model. Type 'exit' to quit.`
`You:`

You can now start chatting with the model.

---

## Appendix: Additional Information

### Where is the Model Saved?

The Hugging Face library automatically downloads and saves the model in a local cache directory. On Ubuntu, the default location is:

**`~/.cache/huggingface/hub`**

* **`~`** is a shortcut for your home directory (e.g., `/home/<user>`).
* **`.cache`** is a hidden directory.

### What is tokenizer.chat_template

* Think of 'tokenizer.chat_template' as a strict formatting recipe that tells the program how to arrange a conversation before showing it to the Llama 3.1 model.

* The model wasn't just trained on raw text, it was trained on text formatted in a very specific way, with special tags to identify who is speaking (the system, the user, or the assistant). If you don't use this exact recipe, the model gets confused and won't perform well, much like a web browser can't display a website correctly if the HTML tags are wrong.

### How to Verify GPU Usage

To confirm the model is running on your GPU:

1.  Open a **new** terminal window while the `chat.py` script is running.
2.  Run the NVIDIA System Management Interface tool:
    ```bash
    nvidia-smi
    ```
3.  You will see a table showing your GPU's status. Look for a `python3` process in the list, and you'll see how much **GPU Memory** it is using and the current **GPU-Utilization**. This confirms the script is using your GPU.