import asyncio
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

async def load_llm(model_id, device, callbacks):
    """
    Load the LLaMA model asynchronously.
    """
    print("Loading the LLaMA model for inference...")
    # Load the LLM using HuggingFacePipeline in a separate thread
    llm = await asyncio.to_thread(
        HuggingFacePipeline.from_model_id,
        model_id,
        task="text-generation",
        device=device,
        pipeline_kwargs={
            "return_full_text": False,
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.7,
        },
        callbacks=callbacks
    )
    llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id
    return llm

async def run_inference_for_question(agent, question):
    """
    Run inference for a single question asynchronously.
    """
    print(f"Running inference for: {question}")
    output = await asyncio.to_thread(agent, question)  # Run inference in a separate thread
    return output

async def test_llm_inference():
    """
    Test the existence of LLM inference using LLaMA 70B, run inference in parallel using async.
    """
    # Step 1: Initialize async tasks
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    callbacks = [StreamingStdOutCallbackHandler()]  # Assuming you have this handler
    test_questions = ["Which is bigger between 9.9 and 9.10?", 
                      "If a warranty is valid for 2 years, can I buy the product on 12 March 2023 and get warranty today on 2 March 2025?",
                      "If a warranty is valid for 2 years, can I buy the product on 12 March 2023 and get warranty today on 24 March 2025?",
                      ]
    prompt_template = PromptTemplate(
        template="You are a helpful assistant. Answer the following questions.\nQ: {question}\nA:",
        input_variables=["question"]
    )

    # Step 2: Load the model in parallel
    llm = await load_llm(model_id, device=3, callbacks=callbacks)

    # Step 3: Prepare agent for inference
    agent = prompt_template | llm

    # Step 4: Run inference for all questions concurrently
    tasks = [run_inference_for_question(agent.batch, question) for question in test_questions]
    results = await asyncio.gather(*tasks)

    # Step 5: Print the output
    print("\nLLM Output:")
    for result in results:
        print(result)

# To run the test_llm_inference function
if __name__ == "__main__":
    asyncio.run(test_llm_inference())

    
