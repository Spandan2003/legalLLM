import pandas as pd

# Replace 'file.jsonl' with the path to your JSONL file
file_path = 'code/legalLLM/hall_detect/factscore_data/labeled/ChatGPT.jsonl'

# Load the JSONL file into a pandas DataFrame
df = pd.read_json(file_path, lines=True)
df['evidence'] = df['annotations']

# Display the DataFrame
df

def extract_evidence(annotations):
    # Check if the input is None and return an empty string if so
    if annotations is None:
        return ""
    
    # Initialize an empty list to store evidence texts
    evidence_texts = []

    # Iterate through each sentence in the annotations
    for sentence in annotations:
        if sentence is not None and sentence['human-atomic-facts']:  # Check if sentence is not None
            # Iterate through each text dictionary in the sentence
            for text in sentence['human-atomic-facts']:
                if text is not None and 'text' in text:  # Ensure 'text' exists in dictionary
                    evidence_texts.append(text['text'])  # Append the text to the list
    
    # Join all evidence texts into a single string
    return " ".join(evidence_texts)
df['evidence'] = df['annotations'].apply(extract_evidence)
df['evidence']

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import  HuggingFacePipeline
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



# Load your dataframe
df = pd.DataFrame({
    'input': ["Who starts first in chess?", "Sample question 2"],  # Replace with your actual questions
    'evidence': ["Chess, a game of intellect and strategy, traces its origins to India around the 6th century AD. It evolved through Persia and eventually reached Europe, where it gained immense popularity. The game is played on an 8x8 checkered board with 16 pieces for each player: a king, a queen, two rooks, two bishops, two knights, and eight pawns. White always moves first. The objective is to checkmate the opponent's king, leaving it in a position where it cannot escape capture. Chess has captivated minds for centuries, offering a complex and ever-evolving challenge. It's a game that transcends language and culture, uniting players from around the world in a battle of wits.", "Sample evidence for question 2"]
})

# Initialize the LLaMA model via Hugging Face Hub
callbacks = [StreamingStdOutCallbackHandler()]
llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # model_id= "microsoft/MiniLM-L12-H384-uncased",
        
        task="text-generation",
        device=0,
        callbacks = callbacks,
        pipeline_kwargs=dict(
            return_full_text=False,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.5,
        ),
    )
llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id
llm_engine = llm

# Step 1: Generate response to each question
generate_response_prompt = PromptTemplate(
    template="Given the question , generate a short single paragraph as a response. Just give the answer. Do not generate anything else: {input}",
    input_variables=["input"]
)
generate_response_chain = generate_response_prompt | llm_engine

# Step 2: Decompose response into atomic facts
decompose_facts_prompt = PromptTemplate(
    template="Given the response: {response}, Please breakdown the sentence into independent facts. Each fact should be on a new line",
    input_variables=["response"]
)
decompose_facts_chain = decompose_facts_prompt | llm_engine

# Step 3: Convert decomposed facts to a list of strings
def split_into_atomic_facts(decomposed_facts):
    # Use a simple splitter or more complex logic as needed
    return str.split(decomposed_facts, sep="\n")
    splitter = RecursiveCharacterTextSplitter(separators=["\n"])
    return splitter.split_text(decomposed_facts)

# Step 4: Classify each fact with respect to evidence
# Classification prompt template for individual facts
classification_prompt = PromptTemplate(
    template=(
        "Classify each fact based on the evidence:\n\n"
        "Evidence: {evidence}\n\n"
        "Facts: {fact}\n\n"
        "Label each fact in Facts as 'Supported', 'Not Supported', or 'Irrelevant'. Give one of these labels as the answer only. Nothing else except the word for each fact in each new line."
        "For instance:"
        "Fact 1"
        "Fact 2"
        "Fact 3"
        "Supported"
        "Not Supported"
        "Supported"


    ),
    input_variables=["evidence", "fact"]
)

classification_chain = classification_prompt | llm_engine

# Function to batch process facts into groups of 10
def batch_process_facts(facts, evidence, batch_size=30):
    batched_facts = [facts[i:i + batch_size] for i in range(0, len(facts), batch_size)]
    all_classified_facts = []

    for batch in batched_facts:
        batch_facts = "\n".join([f"Fact: {fact}" for fact in batch])
        batch_evidence = evidence  # Same evidence applies to all facts in the batch
        response = classification_chain.invoke({"evidence": batch_evidence, "fact": batch_facts})
        classified_batch = process_classification_response(response)
        all_classified_facts.extend(classified_batch)

    return all_classified_facts

# Process the response from classification chain (typically list of fact classifications)
def process_classification_response(response):
    # Assume the response is a list of classifications for each fact in the batch
    # Example: ["Supported", "Not Supported", "Irrelevant", ...]
    return response.split("\n")  # Split the response into individual classifications

# Sequential Chain Combining All Steps
def process_question(input_question, evidence):
    # Generate response
    response = generate_response_chain.invoke({"input": input_question})
    print("Response: ", response)
    print("-x---x---x---x---x---x---x---x---x---x---x---x---x---x---x")
    # Decompose response into atomic facts
    decomposed_facts = decompose_facts_chain.invoke({"response": response})
    print("decomposed_facts: ", decomposed_facts)
    print("-x---x---x---x---x---x---x---x---x---x---x---x---x---x---x")


    # Split into atomic fact list
    atomic_facts = split_into_atomic_facts(decomposed_facts)
    print("atomic_facts: ", atomic_facts)
    print("-x---x---x---x---x---x---x---x---x---x---x---x---x---x---x")


    # Process facts in batches and classify each fact based on evidence
    classified_facts = batch_process_facts(atomic_facts, evidence)
    print("classified_facts: ", classified_facts)
    print("-x---x---x---x---x---x---x---x---x---x---x---x---x---x---x")


    # Return classified facts
    return classified_facts

# Apply the processing pipeline to each question in the dataframe
classified_facts_list = []

for _, row in df.iterrows():
    classified_facts = process_question(row['input'], row['evidence'])
    classified_facts_list.extend(classified_facts)
    break

# Add the classified facts list as a new column in the dataframe
print(classified_facts_list)
count = {'t':0, 's':0, 'ns':0, 'ir':0, 'others':0}

for fact in classified_facts_list:
    count['t'] += 1  # Increment total count for every fact
    if "not supported" in fact.lower():
        count['ns'] += 1
    elif "supported" in fact.lower():
        count['s'] += 1
    elif "irrelevant" in fact.lower():
        count['ir'] += 1
    else:
        count['others'] += 1

# Print the counts
print(count)
print("FACTSCORE = ",count['s']/count['t'])
