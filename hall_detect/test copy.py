import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Create the model and move it to the GPU
model = SimpleModel().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate random data for training
batch_size = 256
input_data = torch.randn(batch_size, 1024).to(device)
target_data = torch.randn(batch_size, 1).to(device)

# Training loop
num_iterations = 2000
print("Starting training...")
for iteration in range(num_iterations):
    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, target_data)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if iteration % 1000 == 0:
        print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}")

print("Training completed.")

from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def test_llm_inference():
    """
    Test the existence of LLM inference using LLaMA 70B.
    """
    # Step 1: Load the LLM
    print("Loading the LLaMA model for inference...")

    # Initialize callbacks to stream outputs
    callbacks = [StreamingStdOutCallbackHandler()]
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Load the LLM using HuggingFacePipeline
    llm = HuggingFacePipeline.from_model_id(
        model_id= model_id,  # Replace with the LLaMA 70B model ID
        task="text-generation",
        device=1,
        pipeline_kwargs={
            "return_full_text": False,  # Only generate new tokens
            "max_new_tokens": 1024,      # Control output length
            "do_sample": True,          # Enable sampling
            "temperature": 0.7,         # Balance creativity and coherence
 #           "stopping_criteria": ["\n", " "],  # Stop after a newline or space
            
        },
        callbacks=callbacks,
    )
    llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id

    # Step 2: Define a test question
    test_question = "Is Mumbai the capital of India?"

    # Step 3: Create a prompt
    prompt_template = PromptTemplate(
        template="You are a helpful assistant. Answer the following question with a Yes or No only. Do not use more than a single word. End the answer with an End of sentence token.:\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["question"]
    )
    #formatted_prompt = prompt_template.format(question=test_question)

    # Step 4: Run inference
    print("\nPerforming inference with "+model_id+" ...")
    agent = prompt_template | llm

    output = agent.invoke(test_question)
    # Step 5: Print the output
    print("\nLLM Output:")
    print(output)

# Run the test
if __name__ == "__main__":
    test_llm_inference()

