import os, torch, logging, numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from evaluate import load

torch.cuda.empty_cache()
rouge_metric = load("rouge")
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
# llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
# llama_tokenizer.pad_token = llama_tokenizer.eos_token
# llama_tokenizer.padding_side = "right"  # Fix for fp16

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=False
# )

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     #quantization_config=quant_config,
#     device_map='auto'
# )
# base_model.config.use_cache = False
# base_model.config.pretraining_tp = 1

# # base_model = PeftModel.from_pretrained(
# #     base_model,
# #     os.path.join("model", "Llama-ft_1")
# #     # os.path.join('./results_modified_sarvam', "checkpoint-5000")
# # )

# llm_dir = r"./code/legalLLM/llama_ft/dataset/Q&A_llm"
# filenames = []
# for filename in os.listdir(llm_dir):
#     filenames.append(llm_dir + "/" + filename)
# # List to store filenames
# train_filenames = []
# test_filenames = []

# # Number of files
# num_files = 51
# 

# Iterate through each file and add it to data_files dictionary
# for i in range(1, 42):
#     filename = f"{i}mod.txt"
#     train_filenames.append(f"mod_files/Q&A/{filename}")
# for i in range(42, 52):
#     filename = f"{i}mod.txt"
#     test_filenames.append(f"mod_files/Q&A/{filename}")

# np.random.seed = 2
# topk = np.random.permutation(np.arange(0, len(filenames)))[:10]

# # Store filenames in data_files dictionary
# data_files = {}
# #data_files["train"] = train_filenames
# data_files["test"] = np.array(filenames)[topk]
# dataset = load_dataset("text", data_files=data_files)


# generated_texts = []
# base_model.eval()
# for example in dataset["test"]:
#     # Tokenize the input query
#     input_text = example
#     input_ids = llama_tokenizer(input_text, return_tensors="pt")["input_ids"].to('cuda')
    
#     # Generate text using the model
#     output = base_model.generate(input_ids)
    
#     # Decode the generated text
#     generated_text = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    
#     # Append the generated text to the list
#     generated_texts.append(generated_text)

# reference_texts = val_data["text"][:1]      #Replace reference column name according to dataset
# rouge_scores = rouge_metric.compute(predictions=generated_texts[0: len(reference_texts)], 
#                                        references=reference_texts,
#                                       use_aggregator=True)

# print("Rouge Score:", rouge_scores)

dt = load_dataset('./code/legalLLM/llama_ft/dataset/Q&A_llm/')
print(dt)