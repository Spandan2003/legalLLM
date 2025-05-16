import pandas as pd

ver = ""
data = pd.read_csv(f'./baselines/hhem/results{ver}.csv')
print(len(data[data["label_consistent"]<=0.5]), len(data), len(data[data["label_consistent"]<=0.5])/len(data))


# Load the dataset
df = data

# Define the hallucination threshold
threshold = 0.5  # Adjust this based on your needs

# Create an empty list to store recombined results
recombined_results = []

# Group by 'chat' and 'turn'
for (chat, turn), group in df.groupby(['chat', 'turn']):
    # Select the first row for context, history, and query (since they are the same across all rows in the group)
    first_row = group.iloc[0]
    
    context = first_row['context']
    history = first_row['history']
    query = first_row['query']
    combined_response = ' '.join(group['response'].tolist())  # Combine all responses
    
    # Filter the group to include only rows with hallucinated label > threshold
    hallucinated_group = group[group['label_hallucinated'] > threshold]
    
    # Combine the responses from the hallucinated rows (if any)
    if not hallucinated_group.empty:
        hallucination = ' '.join(hallucinated_group['response'].tolist())
    else:
        hallucination = ''  # If no hallucinated responses, leave it blank
    
    # Create the recombined row
    recombined_row = {
        'chat': chat,
        'turn': turn,
        'context': context,
        'history': history,
        'query': query,
        'response': combined_response,
        'hallucination': hallucination,
        'label': first_row['label'],  # You can adjust this as needed (e.g., majority voting)
        'label_hallucinated': hallucinated_group['label_hallucinated'].mean() if not hallucinated_group.empty else 0,
        'label_consistent': hallucinated_group['label_consistent'].mean() if not hallucinated_group.empty else 0,
    }
    
    # Append the recombined row to the results list
    recombined_results.append(recombined_row)

# Convert the list of recombined results into a DataFrame
recombined_df = pd.DataFrame(recombined_results)

# Save the recombined dataframe to CSV
recombined_df.to_csv(f'./baselines/hhem/recombined_results{ver}.csv', index=False)

print("Recombined dataset saved successfully.")


threshold = 0.5  # Adjust this based on your needs

# Create an empty list to store recombined results
recombined_results = []

# Group by 'chat' and 'turn'
for (chat, turn), group in data.groupby(['chat', 'turn']):
    # Select the first row for context, history, and query (since they are the same across all rows in the group)
    first_row = group.iloc[0]
    
    context = first_row['context']
    history = first_row['history']
    query = first_row['query']
    combined_response = ' '.join(group['response'].tolist())  # Combine all responses
    
    # Filter the group to include only rows with hallucinated label > threshold
    hallucinated_group = group[group['label_hallucinated'] > threshold]
    
    # To detect continuous hallucinations, we need to identify continuous sequences of hallucinated rows
    hallucination_blocks = []
    current_block = []
    
    # Iterate over the hallucinated responses and group them into continuous blocks
    for idx, row in hallucinated_group.iterrows():
        if not current_block:
            # Start a new block with the first hallucinated row
            current_block.append(row['response'])
        else:
            # Check if current row is part of the same continuous block
            prev_idx = hallucinated_group.index[hallucinated_group.index.get_loc(idx) - 1] if hallucinated_group.index.get_loc(idx) > 0 else None
            if prev_idx is not None and idx == prev_idx + 1:
                # If consecutive, add to current block
                current_block.append(row['response'])
            else:
                # If not consecutive, store the current block and start a new one
                hallucination_blocks.append(current_block)
                current_block = [row['response']]
    
    # If the last block has any responses, append it
    if current_block:
        hallucination_blocks.append(current_block)
    
    # Combine the responses from the hallucinated rows as blocks
    hallucination = hallucination_blocks if hallucination_blocks else []
    
    # Create the recombined row
    recombined_row = {
        'chat': chat,
        'turn': turn,
        'context': context,
        'history': history,
        'query': query,
        'response': combined_response,
        'hallucination': [" ".join(hallucination_part) for hallucination_part in hallucination],  # Store the list of continuous hallucinations
        'label': first_row['label'],  # You can adjust this as needed (e.g., majority voting)
        'label_hallucinated': hallucinated_group['label_hallucinated'].mean() if not hallucinated_group.empty else 0,
        'label_consistent': hallucinated_group['label_consistent'].mean() if not hallucinated_group.empty else 0,
    }
    
    # Append the recombined row to the results list
    recombined_results.append(recombined_row)

# Convert the list of recombined results into a DataFrame
recombined_df = pd.DataFrame(recombined_results)

# Save the recombined dataframe to CSV
recombined_df.to_csv(f'./baselines/hhem/recombined_results_with_blocks{ver}.csv', index=False)