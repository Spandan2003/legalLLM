import pandas as pd
# from lettucedetect.model_evaluator import HallucinationDetector
from lettucedetect.models.inference import HallucinationDetector

data = pd.read_csv('./baselines/processed_dataset.csv')

# Initialize the hallucination detector model
detector = HallucinationDetector(
    method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

# Initialize a list to store results
results = []

# Iterate over each row of the dataset
for idx, row in data.iterrows():
    context = row['context']
    query = row['query']
    response = row['response']
    
    # Get span-level predictions from the HallucinationDetector
    predictions = detector.predict(context=[context], question=query, answer=response, output_format="spans")
    
    # Initialize empty lists to store prediction details
    hallucinated_texts = []
    hallucinated_starts = []
    hallucinated_ends = []
    confidences = []
    
    # If there are hallucinations, extract them
    if predictions:
        for prediction in predictions:
            hallucinated_texts.append(prediction['text'])
            hallucinated_starts.append(prediction['start'])
            hallucinated_ends.append(prediction['end'])
            confidences.append(prediction['confidence'])
    
    # If no hallucinations, leave the lists empty
    if not hallucinated_texts:
        hallucinated_texts = ['No hallucination detected']
        hallucinated_starts = [None]
        hallucinated_ends = [None]
        confidences = [0.0]
    
    # Add the results to the row
    row_result = row.to_dict()
    row_result['hallucinated_texts'] = hallucinated_texts
    row_result['hallucinated_starts'] = hallucinated_starts
    row_result['hallucinated_ends'] = hallucinated_ends
    row_result['hallucination_confidence'] = confidences
    
    # Append the modified row to the results list
    results.append(row_result)

# Convert the results list back to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
results_df.to_csv('./baselines/lettucedetect/results.csv', index=False)

print("Hallucination detection completed and results saved.")