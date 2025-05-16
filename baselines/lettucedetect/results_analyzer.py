import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')

def get_sentence_spans(text):
    """Return list of (start_idx, end_idx, sentence) for each sentence in the text."""
    sentences = sent_tokenize(text)
    spans = []
    current_idx = 0
    for sent in sentences:
        start = text.find(sent, current_idx)
        end = start + len(sent)
        spans.append((start, end, sent))
        current_idx = end
    return spans

def sentence_overlaps_with_any_hallucination(sent_start, sent_end, halluc_spans):
    """Check if sentence span overlaps with any hallucinated span."""
    if halluc_spans[0] == (None, None):
        return False
    for h_start, h_end in halluc_spans:
        if not (sent_end <= h_start or sent_start >= h_end):  # There's some overlap
            return True
    return False

# Load your data
df = pd.read_csv('./baselines/lettucedetect/results.csv')  # replace with actual path

# Prepare list to store hallucination sentences
hallucination_lists = []

for _, row in df.iterrows():
    response = row['response']
    
    # Parse hallucinated span indices
    try:
        halluc_starts = eval(row['hallucinated_starts']) if pd.notna(row['hallucinated_starts']) else []
        halluc_ends = eval(row['hallucinated_ends']) if pd.notna(row['hallucinated_ends']) else []
    except:
        halluc_starts = []
        halluc_ends = []
    
    halluc_spans = list(zip(halluc_starts, halluc_ends))
    
    # Get sentence spans
    sentence_spans = get_sentence_spans(response)
    
    # Extract sentences overlapping with any hallucination
    hallucinated_sentences = [
        sent for (start, end, sent) in sentence_spans
        if sentence_overlaps_with_any_hallucination(start, end, halluc_spans)
    ]
    
    hallucination_lists.append(hallucinated_sentences)

# Add new column to original DataFrame
df['hallucination'] = hallucination_lists

df

df.to_csv('./baselines/lettucedetect/recombined_results.csv', index=False)
print("Length of dataset:", len(df))