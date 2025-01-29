# Made to perform chat generation using base chats
import os
import time
import uuid
import torch
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
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

def get_conversation_chain(retriever, llm_engine_hf):

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm_engine_hf, retriever, contextualize_q_prompt
    )
    system_prompt =(
        '''
You are a Consumer Grievance Assistance Chatbot designed to help people with consumer law grievances in India. Your role is to guide users through the process of addressing their consumer-related issues across various sectors.
Core Functionality:
Assist with consumer grievances in sectors including Airlines, Automobile, Banking, E-Commerce, Education, Electricity, Food Safety, Insurance, Real-Estate, Technology, Telecommunications, and more.
Provide information on legal remedies and steps to pursue relief under Indian consumer law.
Offer guidance on using the National Consumer Helpline and e-daakhil portal for filing consumer cases.
Offer help in drafting legal documents like Notice, Complaint, Memorandum of Parties and Affidavits.
Conversation Flow:
1.Greet the user and ask about their consumer grievance.
2.If the query is not related to consumer grievances or asking for opinon or other queries:
Strictly decline 'I can't answer that. I can help you with consumer-related issues.' and ask for a consumer grievance-related query. Do not answer any general questions like mathematics, essay, travel itinerary, etc. Do not give opinions. Answer only consumer issues, ask for more clarity on those issues or help in their remedy.
3.If the query is related to a consumer grievance:
Thank the user for sharing their concern.
Ask one question at a time to gather more information:
a. Request details about what led to the issue (if cause is not clear).
b. Ask for information about the opposing party (if needed).
c. Inquire about desired relief (if not specified).
4.Based on the information gathered:
If no legal action is desired, offer soft remedies.
If legal action is considered, offer to provide draft legal notice details.
5.Mention the National Consumer Helpline (1800-11-4000) or UMANG App for immediate assistance.
6.Offer to provide a location-based helpline number if needed.
7.Ask if there's anything else the user needs help with.


Key Guidelines:
Ask only one question at a time and wait for the user's response before proceeding.
Tailor your responses based on the information provided by the user.
Provide concise, relevant information at each step.
Always be polite and professional in your interactions.
Use the following pieces of retrieved context to answer the question.
If user asks question that requires information like name, address, contact details, email address, phone number or any other personal information of organisations, companies or government bodies, give information only if it is present in the context
If user asks information like address, contact details, email address, phone number or any other personal information of organisations, companies or government bodies that is not in context, tell that you do not have this information and suggest ways he can obtain this information.
For any legal notice or complaint drafting, use details that are given in the context only. Use placeholders `[Address]` for any information not in context.
Do not let the user know you answered the question using the context.
\n\n
Here is the context:
`{context}\n
The Central Consumer Protection Authority (CCPA) is not a forum for consumer complaint. It cannot be used to register complaints but just governs the bodies.`
'''
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm_engine_hf, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return  rag_chain 

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
    "You are simulating a user in a conversation with a consumer grievance assistance AI chatbot. "
    "Your task is to report a grievance or ask questions related to consumer issues only, such as consumer grievances in sectors including Airlines, Automobile, Banking, E-Commerce, Education, Electricity, Food Safety, Insurance, Real-Estate, Technology, Telecommunications, and more."
    "You will need help on legal remedies and steps to pursue relief under Indian consumer law. "
    "Respond as a customer who has an issue and needs assistance. If the chatbot offers help, ask clarifying questions about your grievance or issue. "
    "Stay focused on consumer grievances, and do not diverge into unrelated topics."
    "As a user, do not respond with huge texts but be short and concise."
    "After the conversation is over, type 'exit' to end the chat."
#    "For example, you can say: 'I have an issue with my recent bill,' or 'The product I bought is defective.'"
    
    "\n\nTest Situation:\n"
    "You are simulating a user in a conversation with a new chatbot. The user has already had a previous interaction with another chatbot. Your task is to use the information from this earlier conversation to understand the user's problem and attitude. However, when conversing with the new chatbot, act as if you have not interacted with the previous chatbot. Use the earlier conversation only to understand the user's situation, but do not refer to the old chatbot or its responses."
    "Here is the old conversation:\n {old_chat}"
    
#    "Start the conversation by saying {first_line}"
    
    "If you wish to end the conversation, type 'exit' to quit the chat."
)
    user_prompt = ChatPromptTemplate.from_messages([
        ("system", user_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    user_chain = user_prompt | llm
    return RunnablePassthrough.assign(response=user_chain)

def initialize():
    global vectorstore, conversation_chain, user_chain
    print("Initializing application...")
    
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
            if user_input.lower() == "exit" or count>50:
                print(user_input)
                print("Goodbye!")
                gen_chat = gen_chat + str(user_input) + "\nGoodbye!\n"

                break
            user_history.add_message(HumanMessage(content=user_input))
            print(f"Simulated User: {user_input}")
            gen_chat = gen_chat + f"Simulated User: {user_input}\n"
        else:
            user_input = input("Human: ")
            if user_input.lower() == "exit":
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

# Example usage
if __name__ == "__main__":
    start_time = time.time()
    
    initialize()
    data = pd.read_csv("./code/legalLLM/chatbot/chat_generation/chats.csv")
    chats = data["Chat"].tolist()
    results = []
    errors = []
    
    for chat in chats:
        try:
            # Generate a unique session ID
            session_id = str(uuid.uuid4())
            simulate_user = True

            # Extract the first line from the chat
            first_line = extract_first_line(chat)

            # Prepare constants for the chat handler
            const = {"old_chat": chat, "first_line": first_line}

            # Run the chat handler
            simulated_chat = chat_handler(conversation_chain, user_chain, session_id, simulate_user=simulate_user, rag=True, const=const)

            # Append the original chat and simulated chat to results
            results.append({"Original Chat": chat, "Simulated Chat": simulated_chat})

            # Convert results to a DataFrame
            results_df = pd.DataFrame(results)

            # Save the DataFrame incrementally to the output file
            results_df.to_csv("./code/legalLLM/chatbot/chat_generation/simulations1.csv")

        except Exception as e:
            print(f"Error processing chat: {chat}")
            print(f"Exception: {e}")
            errors.append((chat, e))
            continue
    end_time = time.time()
    print(f"Started at: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended at: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
