import os, torch, logging
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from tqdm import tqdm
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset

torch.cuda.empty_cache()
#torch.cuda.set_per_process_memory_fraction(1, 0)


# Dataset
# data_name = "wikimedia/wikipedia"
# training_data = load_dataset(data_name, name='20231101.mr', split="train")
# val_data = load_dataset('bigscience-data/roots_indic-mr_wikibooks', split='train')
# training_data = load_dataset('nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions', split='train[:70%]')
# val_data = load_dataset('nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions', split='train[70%:]')
training_data = load_dataset("./code/legalLLM/llama_ft/dataset/Q&A-singleturn.csv")
# print(type(training_data), type(val_data))
# exit()
# num_tokens = 0
# for inst in tqdm(training_data):
#     num_tokens += len(inst['text'].split())

# print('Num of tokens', num_tokens)

# Model and tokenizer names


# df = pd.read_csv('./data/mr_ft.csv')
# dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())

### convert to Huggingface dataset
# training_data = Dataset(pa.Table.from_pandas(df))


base_model_name = "meta-llama/Llama-2-7b-chat-hf"
refined_model = "code/legalLLM/llama_ft/model/Llama-ft_qa"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# # Quantization Config
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=False
# )

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    #quantization_config=quant_config,
    load_in_8bit=True,
    device_map='auto'
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.1,
    r=128,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./code/legalLLM/llama_ft/model",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    optim="paged_adamw_32bit",
    save_steps=1000,
    logging_steps=25,
    learning_rate=3e-4,
    weight_decay=0.1,
    fp16=False,
    bf16=True,
    max_grad_norm=0.9,
    #lr_min=3e-5, # 10% of learning_rate
    max_steps=-1,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="cosine"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    eval_dataset=val_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params,
    max_seq_length=2048,
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)