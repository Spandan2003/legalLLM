import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

ver = ""
# Load the sentence-level dataset
df = pd.read_csv('./baselines/hhem/expanded_dataset.csv')
print(f"Loaded {len(df)} rows from the dataset.")

# Combine premise from context, history, and query
def build_premise(row):
    return f"{row['context']} {row['history']} {row['query']}" # ver ""
    # return f"{row['context']} {row['history']}" # ver "1"

# Create prompt pairs (premise, hypothesis)
df['premise'] = df.apply(build_premise, axis=1)
df['hypothesis'] = df['response']

# Format prompts for the model
prompt_template = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
df['model_input'] = df.apply(lambda row: prompt_template.format(text1=row['premise'], text2=row['hypothesis']), axis=1)

# Initialize the model pipeline
classifier = pipeline(
    "text-classification",
    model='vectara/hallucination_evaluation_model',
    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
    trust_remote_code=True
)

# Run the model and store all outputs
# tqdm gives a nice progress bar for long loops
outputs = []
batch_size = 8  # adjust based on memory
for i in tqdm(range(0, len(df), batch_size), desc="Running model"):
    batch_inputs = df['model_input'].iloc[i:i+batch_size].tolist()
    batch_outputs = classifier(batch_inputs, top_k=None)
    outputs.extend(batch_outputs)
    # print(f"Processed batch {i // batch_size + 1}/{len(df) // batch_size + 1}")

# Store full model outputs (list of dicts) in a new column
df['model_outputs'] = outputs

# Optional: Save to CSV (flattening JSON if needed)
# If you want to flatten the 'model_outputs' for CSV storage:
flat_results = []
for idx, output in enumerate(outputs):
    result = df.iloc[idx].to_dict()
    for item in output:
        result[f"label_{item['label']}"] = item['score']
    flat_results.append(result)

# Convert to DataFrame and save
results_df = pd.DataFrame(flat_results)

# Save the concatenated dataframe to CSV
results_df.to_csv(f'./baselines/hhem/results{ver}.csv', index=False)