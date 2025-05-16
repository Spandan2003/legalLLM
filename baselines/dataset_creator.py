import ast
import pandas as pd

# merged_df2 = pd.read_csv("./baselines/DetectorEval.csv")
merged_df2 = pd.read_csv("./baselines/Redone_dataset_without_annotation.csv")
# Initialize lists
dataset = []

# Iterate over each row in the DataFrame
for i, row in merged_df2.iterrows():
    # Parse the list of dictionaries safely
    try:
        turns = ast.literal_eval(row["Turns"])
        
    except Exception as e:
        print(f"Row {i} parsing error: {e}")
        continue
    
    # Process each dictionary in the list
    for turn_no, turn in enumerate(turns):
        context = turn['context']
        history = turn['history']
        query = turn['query']
        response = turn['response']
        modified_analysis = turn['modified_analysis']

        # Positive case: modified_analysis is an empty string
        if modified_analysis.strip() == '':
            dataset.append({'chat': i, 'turn': turn_no, 'context': context, 'history': history, 'query': query, 'response': response, 'label': 0})
        else:
            dataset.append({'chat': i, 'turn': turn_no, 'context': context, 'history': history, 'query': query, 'response': response, 'label': 1})

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(dataset)
# Save the DataFrame to a CSV file
df.to_csv("./baselines/processed_dataset.csv", index=False)
