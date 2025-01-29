import os
import evaluate
import numpy as np
from typing import List, Union, Iterable
from itertools import zip_longest
from moverscore_v2 import word_mover_score
from collections import defaultdict

def extract_text_from_files(folder1, folder2):
    # Dictionary to store text from both folders
    data = []

    # Get the list of files from both folders (assuming both folders have the same filenames)
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    # Ensure both folders contain the same filenames
    if files1 != files2:
        print("Error: Folder file names don't match.")
        return None
    
    # Loop through the files and extract content
    for filename in files1:
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        
        # Check if both files exist
        if os.path.exists(file1_path) and os.path.exists(file2_path):
            # Read content from folder 1 (ground truth)
            with open(file1_path, 'r', encoding='utf-8') as file1:
                folder1_text = file1.read()

            # Read content from folder 2 (generated)
            with open(file2_path, 'r', encoding='utf-8') as file2:
                folder2_text = file2.read()

            # Store both in the dictionary
            data.append({
                'ground_truth': folder1_text,
                'generated': folder2_text
            })
        else:
            print(f"Error: File {filename} is missing in one of the folders.")
    
    return data

def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score
def corpus_score(data, trace=0):

    sys_stream = []
    ref_streams = []
    for element in data:
        sys_stream.append(element["generated"])
        ref_streams.append(element["ground_truth"])

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]

    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    # fhs = [sys_stream] + ref_streams
    # print(len(sys_stream), len(ref_streams))
    # print(len(fhs))

    corpus_score = 0
    for lines in zip(sys_stream, ref_streams):#zip_longest(*fhs):
        print("st_art", lines[0][:100], lines[0][-100:], lines[1][:100], lines[1][-100:])
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
            
        hypo, refs = lines
        corpus_score += sentence_score(hypo, refs, trace=0)
        
    corpus_score /= len(sys_stream)

    return corpus_score


# Define folder paths
folder1 = './code/legalLLM/chatbot/evaluator/chats_txt_truth'
folder2 = './code/legalLLM/chatbot/evaluator/chats_txt_65'

folder1 = os.path.join(os.getcwd(), folder1)
folder2 = os.path.join(os.getcwd(), folder2)

# Call the function and get the data
file_data = extract_text_from_files(folder1, folder2)[:2]
# file_data = [{"generated":"The world is beautiful", "ground_truth":"The world is beautiful"}, 
#              {"generated":"The world is ending", "ground_truth":"The world is beautiful"}]
print("Moverscore = ", corpus_score(file_data))