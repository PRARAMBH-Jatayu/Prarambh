import torch
from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize FastAPI
app = FastAPI()

# Load the saved model state dictionary
state_dict = torch.load("./model.pt")

# Instantiate a new GPT2LMHeadModel using the same configuration as the trained model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Adjust the size of the model parameters to match the saved state dictionary
state_dict['transformer.wte.weight'] = state_dict['transformer.wte.weight'][:model.config.vocab_size, :]
state_dict['lm_head.weight'] = state_dict['lm_head.weight'][:model.config.vocab_size, :]

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the device to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()


# Define the root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Conversational Chatbot API for Healthcare and Insurence!"}


# Define the inference endpoint
@app.post("/chat/")
def infer(text: str):
    # Preprocess the input text
    input_text = "<startofstring> " + text + " <bot>: "
    input_text = tokenizer(input_text, return_tensors="pt")
    X = input_text["input_ids"].to(device)
    attention_mask = input_text["attention_mask"].to(device)

    # Generate the output using the model
    output = model.generate(X, attention_mask=attention_mask)
    output_text = tokenizer.decode(output[0])

    return {"input": text, "output": output_text}


# Run the API with uvicorn server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

