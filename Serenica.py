###############################################################################
# serenica.py
# An all-in-one script that (1) trains or loads a classification model,
# (2) trains or loads a GPT model, (3) runs a Flask webserver
# that uses classification + GPT to respond to user queries.
###############################################################################

import argparse
import os
import re
import nltk
import pandas as pd
import torch
import json

from flask import Flask, render_template_string, request
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

###############################################################################
# 1. Text Preprocessing
###############################################################################
def preprocess_text(text, do_lower=True, remove_punc=True, do_lemmatize=False):
    if do_lower:
        text = text.lower()
    if remove_punc:
        text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)
    sw = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in sw]

    if do_lemmatize:
        lemmatizer = WordNetLemmatizer()
        filtered = [lemmatizer.lemmatize(token) for token in filtered]

    return " ".join(filtered)

###############################################################################
# 2. Classification Model & Dataset
###############################################################################
class ClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_classifier(
    data_path="classification_data.csv",
    model_name="bert-base-uncased",
    output_dir="cls_model_output",
    num_labels=3,
    epochs=3,
    batch_size=8
):
    print(f"[INFO] Training BERT classifier from {data_path}")
    df = pd.read_csv(data_path).dropna()

    # Preprocess each text
    df["text"] = df["text"].apply(lambda x: preprocess_text(x))

    # Shuffle & split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.9 * len(df))
    train_df = df[:split_idx].copy()
    eval_df = df[split_idx:].copy()

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_dataset = ClassificationDataset(train_df, tokenizer)
    eval_dataset = ClassificationDataset(eval_df, tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    if torch.cuda.is_available():
        model.cuda()

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        weight_decay=0.01,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        push_to_hub=False
    )

    # We define a custom Trainer data collator
    def collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Classification model saved to {output_dir}")

def load_classification_model(model_path="cls_model_output"):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print(f"[INFO] Loaded classification model from {model_path}")
    return model, tokenizer

def predict_label(model, tokenizer, text):
    processed = preprocess_text(text)
    inputs = tokenizer(processed, return_tensors='pt', truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).cpu().item()
    return pred

###############################################################################
# 3. GPT Fine-Tuning
###############################################################################
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = str(self.data.loc[idx, 'prompt'])
        response = str(self.data.loc[idx, 'response'])

        text = f"Question: {prompt}\nAnswer: {response}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }

def train_gpt(
    data_path="counsel_prepared.csv", 
    model_name="distilgpt2",
    output_dir="gpt_model_output",
    epochs=2,
    batch_size=2
):
    print(f"[INFO] Training GPT model from {data_path}")
    df = pd.read_csv(data_path)
    df.dropna(subset=['prompt','response'], inplace=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.9 * len(df))
    train_df = df[:split_idx].copy()
    eval_df = df[split_idx:].copy()

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.cuda()

    train_dataset = QADataset(train_df, tokenizer)
    eval_dataset = QADataset(eval_df, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        weight_decay=0.01,
        warmup_steps=100,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] GPT model saved to {output_dir}")

def load_gpt_model(model_path="gpt_model_output"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print(f"[INFO] Loaded GPT model from {model_path}")
    return model, tokenizer

def generate_response(cls_model, cls_tokenizer, gpt_model, gpt_tokenizer,
                      user_input, label_map=None, conversation=None):
    """
    1) Classify the user_input with BERT model => get numeric label => label string
    2) Incorporate label + conversation into GPT prompt => generate answer
    3) Return the GPT answer
    """
    # 1) Predict classification label
    pred_label = predict_label(cls_model, cls_tokenizer, user_input)
    label_str = str(pred_label)
    if label_map and pred_label in label_map:
        label_str = label_map[pred_label]

    # 2) Build a conversation-based prompt. 
    #    We'll add the classification label at the start.
    # conversation is a list of dict: {"role": "user"/"bot", "content": "..."}
    # We'll transform that into text lines, then add the new user input.
    conv_text = f"Topic: {label_str}\n"
    if conversation:
        for turn in conversation:
            if turn["role"] == "user":
                conv_text += f"User: {turn['content']}\n"
            else:
                conv_text += f"Bot: {turn['content']}\n"
    conv_text += f"User: {user_input}\nBot:"

    # 3) GPT generate
    input_ids = gpt_tokenizer.encode(conv_text, return_tensors='pt')
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        output_ids = gpt_model.generate(
            input_ids,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=gpt_tokenizer.eos_token_id,
            do_sample=True
        )
    generated_text = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # We'll parse out everything after the last "Bot:"
    if "Bot:" in generated_text:
        # split from the right, to handle earlier lines
        answer = generated_text.rsplit("Bot:", 1)[-1].strip()
    else:
        answer = generated_text

    return answer

###############################################################################
# 4. Flask Webserver
###############################################################################

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Serenica Therapy Chatbot</title>
    <style>
      body {
        margin: 0; padding: 0;
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
        animation: fadeIn 1.2s ease-in;
      }
      @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
      }
      .chat-container {
        max-width: 600px; margin: 50px auto;
        padding: 20px; background-color: #fff;
        border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.1);
      }
      h2 {
        text-align: center;
      }
      #chat-window {
        border: 1px solid #ddd; border-radius: 5px;
        padding: 10px; height: 300px; overflow-y: auto;
        margin-bottom: 10px;
      }
      .user-msg {
        text-align: right; margin: 5px 0; color: #333;
      }
      .bot-msg {
        text-align: left; margin: 5px 0;
        background-color: #e8f5e9; display: inline-block;
        padding: 8px 10px; border-radius: 5px;
      }
      .input-container {
        display: flex; gap: 10px;
      }
      #user_input {
        flex: 1;
        padding: 10px; border-radius: 5px;
        border: 1px solid #ddd;
      }
      #send-btn, #reset-btn {
        padding: 10px 20px; border: none;
        border-radius: 5px; cursor: pointer;
      }
      #send-btn {
        background-color: #007bff; color: #fff;
      }
      #reset-btn {
        background-color: #dc3545; color: #fff;
      }
    </style>
</head>
<body>
  <div class="chat-container">
    <h2>Serenica Therapy Chatbot</h2>
    <div id="chat-window"></div>
    <div class="input-container">
      <input type="text" id="user_input" placeholder="Type your message..." />
      <button id="send-btn">Send</button>
      <button id="reset-btn">Reset</button>
    </div>
  </div>

  <script>
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user_input');
    const sendBtn = document.getElementById('send-btn');
    const resetBtn = document.getElementById('reset-btn');

    function appendMessage(content, sender) {
      const msgDiv = document.createElement('div');
      if (sender === 'user') {
        msgDiv.classList.add('user-msg');
        msgDiv.textContent = "You: " + content;
      } else {
        msgDiv.classList.add('bot-msg');
        msgDiv.textContent = "Bot: " + content;
      }
      chatWindow.appendChild(msgDiv);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return;
      appendMessage(text, 'user');
      userInput.value = "";

      try {
        const res = await fetch('/api/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_input: text })
        });
        const data = await res.json();
        if (data.response) {
          appendMessage(data.response, 'bot');
        } else {
          appendMessage("Error: no response field", 'bot');
        }
      } catch (err) {
        console.error(err);
        appendMessage("Could not reach server", 'bot');
      }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        sendMessage();
      }
    });

    resetBtn.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/reset', { method: 'POST' });
        if (res.ok) {
          chatWindow.innerHTML = '';
          appendMessage("Conversation reset.", "bot");
        }
      } catch (err) {
        console.error(err);
      }
    });
  </script>
</body>
</html>
"""

app = Flask(__name__)

# Conversation memory: list of {"role": "user"/"bot", "content": "..."}
CONVERSATION_HISTORY = []

# We also define a label map for classification
LABEL_MAP = {0: "anxiety", 1: "depression", 2: "relationship"}  # example

# Global references to loaded models
CLS_MODEL = None
CLS_TOKENIZER = None
GPT_MODEL = None
GPT_TOKENIZER = None

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/api/message', methods=['POST'])
def api_message():
    global CONVERSATION_HISTORY
    data = request.get_json()
    user_text = data.get("user_input", "")

    # Generate bot response using classification + GPT
    bot_text = generate_response(
        CLS_MODEL, CLS_TOKENIZER,
        GPT_MODEL, GPT_TOKENIZER,
        user_text,
        label_map=LABEL_MAP,
        conversation=CONVERSATION_HISTORY
    )

    # Add to conversation
    CONVERSATION_HISTORY.append({"role": "user", "content": user_text})
    CONVERSATION_HISTORY.append({"role": "bot", "content": bot_text})

    return {"response": bot_text}

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = []
    return {"status": "reset"}

###############################################################################
# 5. MAIN LOGIC: Argument Parsing, (Re)Training, Loading, Running
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-classifier", action="store_true",
                        help="Train the BERT classifier using classification_data.csv")
    parser.add_argument("--train-gpt", action="store_true",
                        help="Train the GPT model using counsel_prepared.csv")
    parser.add_argument("--run-server", action="store_true",
                        help="Run the Flask server (default if no args).")
    args = parser.parse_args()

    # Potentially train classifier
    if args.train_classifier:
        train_classifier(data_path="classification_data.csv", num_labels=3)
    # Potentially train GPT
    if args.train_gpt:
        train_gpt(data_path="counsel_prepared.csv")

    # If neither training arg was provided, or if user also wants to run server:
    if (not args.train_classifier and not args.train_gpt) or args.run_server:
        # 1) Load classification model
        global CLS_MODEL, CLS_TOKENIZER
        CLS_MODEL, CLS_TOKENIZER = load_classification_model("cls_model_output")
        # 2) Load GPT model
        global GPT_MODEL, GPT_TOKENIZER
        GPT_MODEL, GPT_TOKENIZER = load_gpt_model("gpt_model_output")

        # 3) Run Flask
        print("[INFO] Starting Flask server at http://127.0.0.1:5000")
        app.run(debug=True)

if __name__ == "__main__":
    main()
