import pandas as pd
import nltk
# nltk.download('punkt')  # Ensure that the tokenizer models are downloaded
# nltk.download('punkt_tab')  # Ensure that the tokenizer models are downloaded

# Load the dataset from a CSV file (or use another format depending on what you saved it as)
df = pd.read_csv('./baselines/processed_dataset.csv')

# Function to split response into sentences
def split_into_sentences(response):
    return nltk.sent_tokenize(response)

# Apply the function to the 'response' column to create a new 'sentences' column
df['sentences'] = df['response'].apply(split_into_sentences)

# Create a list to store the expanded rows
expanded_data = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Get the sentences and other columns
    sentences = row['sentences']
    
    # Iterate over the sentences and create a new row for each one
    for sentence_no, sentence in enumerate(sentences):
        expanded_data.append({
            'chat': row['chat'],
            'turn': row['turn'],
            'context': row['context'],
            'history': row['history'],
            'query': row['query'],
            'response': sentence,  # Use the sentence instead of the full response
            'label': row['label'],
            'sentence_no': sentence_no  # Sentence number
        })

    if index%10==0:
        print(f"{index}/{len(df)} done")

# Convert the list of expanded data into a new DataFrame
expanded_df = pd.DataFrame(expanded_data)

# Save the expanded dataset to a new CSV file
expanded_df.to_csv('./baselines/hhem/expanded_dataset.csv', index=False)