# Made to perform chat generation using base chats
from chatbots.app_hall0 import get_conversation_chain as get_conversation_chain0
from chatbots.app_hall1 import get_conversation_chain as get_conversation_chain1
from chatbots.app_hall2 import get_conversation_chain as get_conversation_chain2
from chatbots.app_hall3 import get_conversation_chain as get_conversation_chain3
from chatbots.app_hall4 import get_conversation_chain as get_conversation_chain4
import os
import time
import uuid
import torch
import pandas as pd
from PyPDF2 import PdfReader
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
import os
import uuid
import json
from flask_cors import CORS 
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# langchain.debug = True
print("Device used is ", device, ": ")

# Specify the folder containing your PDF files
PDF_FOLDER = '/data/nlp/spandan/code/legalLLM/chatbot/general_chatbot/dataset/rag'
BASE_URL = '/consumer_chatbot'
# Global variables
vectorstore = None
consumer_conversation_chain = None
general_conversation_chain = None
### Statefully manage chat history ###
consumer_store = {}
general_store = {}


def get_consumer_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in consumer_store:
        consumer_store[session_id] = ChatMessageHistory()
        # Add the initial AI message to the chat history
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        consumer_store[session_id].add_message(AIMessage(content=initial_message))
    return consumer_store[session_id]

def get_general_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in general_store:
        general_store[session_id] = ChatMessageHistory()
        # Add the initial AI message to the chat history
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        general_store[session_id].add_message(AIMessage(content=initial_message))
    return general_store[session_id]

def get_pdf_text(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1",model_kwargs={"device": device})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_llm():
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        task="text-generation",
        device=0,
        callbacks = callbacks,
        pipeline_kwargs=dict(
            return_full_text=False,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.5,
        ),
    )
    llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id
    llm_engine_hf = ChatHuggingFace(llm=llm)

    return llm_engine_hf

def get_llm():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    callbacks = [StreamingStdOutCallbackHandler()]
    llm_engine_hf = ChatOpenAI(
        # model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        openai_api_key="EMPTY",
        openai_api_base="http://172.17.0.1:8080/v1/",
        max_tokens=1024,
        temperature=0.5,
        )
    return llm_engine_hf

def get_user_chain(llm):
    # Create user chain prompt
#     user_system_prompt = (
#     "You are simulating a user in a conversation with a consumer grievance assistance AI chatbot. "
#     "Your task is to report a grievance or ask questions related to consumer issues only, such as billing problems, faulty products, or service complaints. "
#     "Respond as a customer who has an issue and needs assistance. If the chatbot offers help, ask clarifying questions about your grievance or issue. "
#     "Stay focused on consumer grievances, and do not diverge into unrelated topics. "
#     "For example, you can say: 'I have an issue with my recent bill,' or 'The product I bought is defective.'"
    
#     "\n\nTest Situation:\n"
#     "You are a customer who received a bill that seems to have an incorrect charge for a service. You want to inquire about the charge. "
#     "Start the conversation by saying something like 'I have a question about a charge on my last bill.'"
    
#     "If you wish to end the conversation, type 'exit' to quit the chat."
# )
    user_system_prompt = (
#     "You are simulating a user in a conversation with a consumer grievance assistance AI chatbot. "
#     "Your task is to report a grievance or ask questions related to consumer issues only, such as consumer grievances in sectors including Airlines, Automobile, Banking, E-Commerce, Education, Electricity, Food Safety, Insurance, Real-Estate, Technology, Telecommunications, and more."
#     "You will need help on legal remedies and steps to pursue relief under Indian consumer law. "
#     "Respond as a customer who has an issue and needs assistance. If the chatbot offers help, ask clarifying questions about your grievance or issue. "
#     "Stay focused on consumer grievances, and do not diverge into unrelated topics."
#     "As a user, do not respond with huge texts but be short and concise."
#     "After the conversation is over, type 'exit' to end the chat."
    
#     "\n\nTest Situation:\n"
#     "You are simulating a user in a conversation with a new chatbot. The user has already had a previous interaction with another chatbot. Your task is to use the information from this earlier conversation to understand the user's problem and attitude. However, when conversing with the new chatbot, act as if you have not interacted with the previous chatbot. Use the earlier conversation only to understand the user's situation, but do not refer to the old chatbot or its responses."
#     "Here is the old conversation:\n {old_chat}"
    
# #    "Start the conversation by saying {first_line}"
    
#     "If you wish to end the conversation, type 'exit' to quit the chat."
"""You are simulating a user in a conversation with a consumer grievance assistance AI chatbot called Nyaya. Your task is to as a user report a grievance or ask questions related to consumer issues only, such as consumer grievances in sectors including Airlines, Automobile, Banking, E-Commerce, Education, Electricity, Food Safety, Insurance, Real-Estate, Technology, Telecommunications, and more.

The framework of chatting:
1. At the initial stage, you will ask Nyaya for legal remedies or guidance on how to proceed with your grievance under Indian consumer law. 
2. Once Nyaya provides you with the necessary information, you may request a notice or complaint letter to be drafted, if needed. 3. 3. After this, the conversation will be concluded, and you will exit the chat.
4. The user should talk in short conversations over multiple turns. In other words he should ask questions regarding things he is confused about not at same time but one after the other. 
5. You do not have any legal expertise so do not suggest any steps to be taken and passively answer any questions Nyaya asks and choose the advice given by Nyaya.

Test Situation:
You are simulating a user in a conversation with Nyaya. The user has already had a previous interaction with another consumer grievance assistance AI chatbot. You need to simulate a fresh conversation, but you can use the details from the Old Chat to understand the user's situation and grievance.

However, do not reference the previous chatbot or the prior conversation. You should behave as if starting a new conversation without any memory of the old chat except regarding the user's situation. Focus only on the user's current grievance, knowledge and situation. You are not to provide any legal advice, you have come here to seek help and have no knowledge about the legal domain.

Here is the information from the old conversation (user's situation) for your reference, Old Chat:
 {old_chat}

Start the conversation by asking for assistance with your grievance, based on the information you learned from the previous interaction, without mentioning the old chat. Once you receive the guidance you need, request a draft of a notice or complaint letter, if applicable, and then reply back with only 'exit' to end the chat."""
)
    user_prompt = ChatPromptTemplate.from_messages([
        ("system", user_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    user_chain = user_prompt | llm
    return RunnablePassthrough.assign(response=user_chain)

def initialize(get_conversation_chain):
    global vectorstore, conversation_chain, user_chain, llm
    print("Initializing application... with ", get_conversation_chain)
    
    raw_text = get_pdf_text(PDF_FOLDER)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever()
    
    llm = get_llm()
    conversation_chain = get_conversation_chain(retriever, llm)
    user_chain = get_user_chain(llm)
    print("Application initialized.")

'''# Initialize global variables
chat_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
        # Add an initial AI message
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        chat_histories[session_id].add_message(AIMessage(content=initial_message))
    return chat_histories[session_id]

def chat_handler(conversation_chain, user_chain, session_id, rag=True):
    history = get_session_history(session_id)
    print(f"Session ID: {session_id}")
    print("Chat history:", [
        {"role": "AI" if isinstance(msg, AIMessage) else "Human", "content": msg.content}
        for msg in history.messages
    ])

    print("Chatbot: ", history[0]['content'])
    
    while True:
        user_input = input("Human: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        history.add_message(HumanMessage(content=user_input))
        # print(datetime.now())
        # print("User input:", user_input)
        
        if rag:
            conversational_rag_chain = RunnableWithMessageHistory(
                conversation_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        else:
            conversational_rag_chain = RunnableWithMessageHistory(
                conversation_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )
        
        parser = StrOutputParser()
        chain = conversational_rag_chain
        config = {"configurable": {"session_id": session_id}}
        
        response = chain.invoke({"input": user_input}, config)
        
        if rag:
            # print("Response:", response)
            formatted_response = response['answer'].replace('\n', '\n')
        else:
            # print("Response:", response)
            formatted_response = response.content.replace('\n', '\n')
        
        history.add_message(AIMessage(content=formatted_response))
        print(f"Chatbot: {formatted_response}")'''

# Initialize global variables
chat_histories = {}

def get_chatbot_session_history(session_id: str) -> ChatMessageHistory:
    """Get or initialize the chatbot's session history."""
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        chat_histories[session_id].add_message(AIMessage(content=initial_message))
    return chat_histories[session_id]

def get_user_session_history(session_id: str) -> ChatMessageHistory:
    """Get or initialize the user LLM's session history."""
    user_session_id = f"user-{session_id}"
    if user_session_id not in chat_histories:
        chat_histories[user_session_id] = ChatMessageHistory()
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        chat_histories[user_session_id].add_message(HumanMessage(content=initial_message))
    return chat_histories[user_session_id]

def chat_handler(chatbot_chain, user_chain, session_id, simulate_user=False, rag=True, const = {"old_chat":"", "first_line":""}):
    """Handles the chat between the chatbot and a user (simulated or real)."""
    gen_chat = ""

    chatbot_history = get_chatbot_session_history(session_id)
    user_history = get_user_session_history(session_id)

    print(f"Session ID: {session_id}")
    print("---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---")
    print(f"Chatbot: {chatbot_history.messages[0].content}")
    gen_chat = gen_chat + f"Session ID: {session_id}\n"
    gen_chat = gen_chat + f"Chatbot: {chatbot_history.messages[0].content}\n"



    # Create RunnableWithMessageHistory for both chatbot and user chains
    chatbot_runnable = RunnableWithMessageHistory(
        chatbot_chain,
        get_chatbot_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer" if rag else "response",
    )
    user_runnable = RunnableWithMessageHistory(
        user_chain,
        get_user_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="response",
    )
    first = True
    count = 1

    while True:
        if simulate_user:
            if first:
                user_input = const['first_line']
                first = False
            else:
                # Use the user LLM to generate the input
                user_response = user_runnable.invoke({"input": chatbot_history.messages[-1].content, "old_chat":const["old_chat"]}, {"configurable": {"session_id": session_id}})
                user_input = user_response['response'].content
                count+=1
            if any([exit_word in user_input.lower() for exit_word in ["exit"]]) or count>25:
                print(user_input)
                print("Goodbye!")
                print("Count is ", count)
                gen_chat = gen_chat + str(user_input) + "\nGoodbye!\n"

                break
            user_history.add_message(HumanMessage(content=user_input))
            print(f"Simulated User: {user_input}")
            gen_chat = gen_chat + f"Simulated User: {user_input}\n"
        else:
            user_input = input("Human: ")
            if any([exit_word in user_input.lower() for exit_word in ["exit", "quit", "goodbye"]]):
                print("Goodbye!")
                gen_chat = gen_chat + "\nGoodbye!\n"
                break
            user_history.add_message(HumanMessage(content=user_input))

        # Add user input to chatbot's history
        chatbot_history.add_message(HumanMessage(content=user_input))
        
        # Generate chatbot's response
        chatbot_response = chatbot_runnable.invoke({"input": user_input}, {"configurable": {"session_id": session_id}})
        formatted_response = (
            chatbot_response['answer'].replace('\n', '\n')
            if rag else chatbot_response.content.replace('\n', '\n')
        )

        # Add chatbot response to both histories
        chatbot_history.add_message(AIMessage(content=formatted_response))
        user_history.add_message(AIMessage(content=formatted_response))

        print(f"Chatbot: {formatted_response}")
        gen_chat = gen_chat + f"Chatbot: {formatted_response}\n"
    
    return gen_chat


import re

def extract_first_line(conversation_text):
    """
    Extracts the first user's input in a conversation between 'User:' and 'AI:'.
    Assumes the conversation alternates between 'User:' and 'AI:'.

    Parameters:
        conversation_text (str): The full conversation as a string.

    Returns:
        str: The user's first line of input.
    """
    # Regex to find the first user input, which is between 'User:' and the next 'AI:'
    match = re.search(r'Human:\s*(.*?)\s*AI:', conversation_text, re.DOTALL)
    if match:
        # Return the captured first line
        return match.group(1).strip()
    else:
        # If the structure is unexpected, return a fallback message
        return ""

def get_user_summarizer(llm):
    system_prompt = """You are provided with an old conversation between a Human (the user) and an AI. Your task is to summarize the user's situation based purely on their own words. Focus only on the user’s grievance, their doubts, their perspective, and their knowledge. Do not mention anything related to the AI’s responses, advice, or suggestions. Ignore all legal advice, notices, complaints, and any suggestions given by the AI in the conversation.

The conversation is as follows:
{old_chat}

Output the following:
A brief turnwise description user's grievance, situation, and any opinions or feelings expressed by the user about the situation. It should be in the format of 
Turn 1:
Turn 2:
...

Instructions for the output:
1. Do not include any AI responses or advice in your summary.
2. Focus only on the user's perspective and their grievance.
3. Ensure to keep a clear note of what things the user does not know and the choices the user makes. Specifically note whether he knew the choices or it was something the bot told him about
Make sure your response only includes the user's perspective. Do not refer to the AI's advice or responses.
"""

    summarizer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", ""),
    ])
    summarizer_chain = summarizer_prompt | llm
    return summarizer_chain

# Example usage
if __name__ == "__main__":
    start_time = time.time()
    end_time_list = [start_time]
    
    data = pd.read_csv("./chatbot/chat_generation/chats.csv")
    chats = data["Chat"].tolist()
    

    chain_callers = [get_conversation_chain0, get_conversation_chain1, get_conversation_chain2, get_conversation_chain3, get_conversation_chain4]

    for i, get_conversation_chain in enumerate(chain_callers[0:1], start=0):
        print(f"Running conversation chain {i}")
        initialize(get_conversation_chain)
        summarizer_chain = get_user_summarizer(llm)
        results = []
        errors = []
        for chat_no, chat in enumerate([chats[26]]):
            try:
                # Generate a unique session ID
                print("START CHAT GENERATION ", chat_no)
                session_id = str(uuid.uuid4())
                simulate_user = True

                # Extract the first line from the chat
                first_line = extract_first_line(chat)
                print("Original Chat:\n", chat)
                # chat.replace("AI:", "Nyaya:").replace("Human:", "Chatbot:")
                # Prepare constants for the chat handler
                chat_sum = summarizer_chain.invoke({"old_chat": chat}).content
                const = {"old_chat": chat_sum, "first_line": first_line}
                print("---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---")
                print("Summarized Chat:\n", chat_sum)
            
                print("---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---")
                # Run the chat handler
                simulated_chat = chat_handler(conversation_chain, user_chain, session_id, simulate_user=simulate_user, rag=True, const=const)
                
                # Append the original chat and simulated chat to results
                results.append({"Original Chat": chat, "Summarized Chat":chat_sum, "Simulated Chat": simulated_chat})

                # Convert results to a DataFrame
                results_df = pd.DataFrame(results)

                # Save the DataFrame incrementally to the output file
                # results_df.to_csv(f"./chatbot/chat_generation/results2/app_halli{i}.csv")

            except Exception as e:
                print(f"Error processing chat: {chat}")
                print(f"Exception: {e}")
                errors.append((chat_no, chat, e))
                continue
        print(f"List of Errors for Conversation Chain {i}: ")
        for error in errors:
            print("Chat is ", error[0], "\nError is ", error[2], "\nChat is ", error[1])
        end_time = time.time()
        end_time_list.append(end_time)
        print(f"Started at: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ended at: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Started at: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    for i in range(4):
        print(f"App_hall{i+1} Ended at: {datetime.fromtimestamp(end_time_list[i+1]).strftime('%Y-%m-%d %H:%M:%S')}      Time taken: {end_time_list[i+1] - end_time_list[i]:.2f} seconds")
    print(f"Time taken: {end_time_list[-1] - start_time:.2f} seconds")

