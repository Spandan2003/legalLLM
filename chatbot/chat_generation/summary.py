import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import pandas as pd

def summarize_chats(chat_samples, llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda"):
    """
    Generate summaries for sampled chats using a specified LLM.

    Parameters:
        chat_samples (pd.Series): The sampled chat data.
        llm_model (str): Hugging Face LLM model name.
        device (str): Device for computation ("cuda" or "cpu").

    Returns:
        list: Summaries for each sampled chat.
    """
    # Load the LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id=llm_model,
        task="text-generation",
        device=0 if device == "cuda" else -1,  # Use GPU (0) or CPU (-1)
        pipeline_kwargs={
            "return_full_text": False,
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
        },
    )
    
    # Define the summarization prompt template
    summarization_prompt_template = PromptTemplate(
        template="You are a summarizer. Given the following chat, provide a concise summary:\n\nChat:\n{chat}\n\nSummary:",
        input_variables=["chat"]
    )
    agent = summarization_prompt_template | llm
    summaries = agent.batch([{"chat": chat} for chat in chat_samples])

    return summaries


data = pd.read_csv("./code/legalLLM/hall_detect/data1_modified.csv")
data = data["Chat"]
data.dropna(inplace=True)
start_letter = data.iloc[0][:4]
data = data[data.str.startswith(start_letter)]
data = data.sample(n=30, random_state=42)
data.reset_index(inplace=True, drop=True)

# Generate summaries for the sampled chats
#summaries = summarize_chats(data)

# Combine sampled chats and their summaries
result = pd.DataFrame({"Chat": data.values})#, "Summary": summaries})
result.to_csv("./code/legalLLM/chatbot/chat_generation/chats.csv")