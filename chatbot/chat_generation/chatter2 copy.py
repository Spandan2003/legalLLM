# Made to perform chat generation using base chats
import os
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
    "You will need on legal remedies and steps to pursue relief under Indian consumer law. "
    "Respond as a customer who has an issue and needs assistance. If the chatbot offers help, ask clarifying questions about your grievance or issue. "
    "Stay focused on consumer grievances, and do not diverge into unrelated topics. "
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
    chatbot_history = get_chatbot_session_history(session_id)
    user_history = get_user_session_history(session_id)

    print(f"Session ID: {session_id}")
    print(f"Chatbot: {chatbot_history.messages[0].content}")

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
                break
            user_history.add_message(HumanMessage(content=user_input))
            print(f"Simulated User: {user_input}")
        else:
            user_input = input("Human: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
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
    # initialize()
    data = pd.read_csv("./code/legalLLM/chatbot/chat_generation/chats.csv")
    chats = data["Chat"].tolist()
    
    session_id = str(uuid.uuid4())
    simulate_user = True
    chat = '''AI: Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you.


Human: My Uber showed my 250 as the rate while booking but charged me 400. I tried using the app's complaint mechanism and they are saying the cab probably took a longer route but it took the same way I do every day. what should I do?

AI: Thank you for sharing your concern with me. I'm here to help you with your consumer grievance.

To better understand the situation, can you please tell me what led to this issue? Was it a one-time incident or a recurring problem with Uber's pricing?

Human: It was a one time incident but it is a route i take everyday. 

AI: So it seems like the incident occurred on a route you regularly take, and the app initially quoted a lower fare but charged you a higher amount.

I would recommend filing a formal complaint with Uber through their official channels, as you've already done. However, if you're not satisfied with their response, you can also consider taking the following steps:

1.  Check your Uber account: Review your account history to see if there are any similar incidents in the past where the fare was higher than quoted.
2.  Provide evidence: If you have any evidence, such as screenshots or GPS data, that shows the route taken by the driver was the same as usual, you can attach it to your complaint.
3.  Escalate the issue: If Uber's customer support doesn't resolve your issue, you can escalate it to their higher authorities or their social media handles.
4.  Consider filing a complaint with the National Consumer Helpline (1800-11-4000) for assistance and guidance on the next steps.

Additionally, you may also want to check if there are any consumer courts or consumer protection agencies in your area that can help you with your grievance.

Is there anything else you'd like to know or any further assistance you need?

Human: are there any consumer courts in Tripura?


AI: According to the Consumer Protection Act, 2019, consumer courts operate at three levels: District Commission, State Commission, and National Commission.


In Tripura, the State Consumer Dispute Redressal Commission (SCDRC) is located in Agartala. The address is:


State Consumer Dispute Redressal Commission,
Tripura,
Agartala.


You can also contact them at their phone number: 0381-232-0325 or 0381-232-0326.


Additionally, you can also file a complaint with the National Consumer Helpline (1800-11-4000) for assistance and guidance on the next steps.


If you need help with drafting a complaint or have any further questions, feel free to ask.


Human: yes can you draft a complaint for uber


AI: Here's a draft complaint you can use as a template:


**Complaint against Uber**


**To,**
Uber India Private Limited,
[Address],


**Subject: Overcharging and Misleading Fare Calculation on [Date]**


**Complaint Details:**


*   Date of Incident: [Date]
*   Time of Incident: [Time]
*   Location of Incident: [Location]
*   Route Taken: [Route Taken]
*   Fare Quoted: ₹250
*   Fare Charged: ₹400


**Description of Incident:**


On [Date], I booked an Uber ride from [Pickup Location] to [Drop Location] using the Uber app. The estimated fare was ₹250, but when I reached my destination, I was charged ₹400. I have been using this route regularly, and the fare has always been ₹250. However, on this occasion, I was overcharged by ₹150.


**Evidence:**


I have attached screenshots of the ride details, including the estimated fare and the actual fare charged. I have also included GPS data that shows the route taken by the driver was the same as usual.


**Demand:**


I request that Uber takes immediate action to rectify this issue and provides a refund of the excess amount charged, i.e., ₹150. I also request that Uber takes necessary steps to ensure that their fare calculation system is accurate and transparent.


**Signature:**


[Your Signature]


**Contact Information:**


*   Name: [Your Name]
*   Email: [Your Email]
*   Phone Number: [Your Phone Number]


Please note that this is a draft, and you should modify it according to your specific situation. Also, make sure to keep a copy of the complaint for your records.


Before filing the complaint, please review it carefully and attach any supporting evidence, such as screenshots and GPS data.


Do you want me to help you with anything else?


Human: This would be all


AI: It was a pleasure assisting you with your consumer grievance. I hope you get a satisfactory resolution from Uber and the consumer courts.


If you have any other consumer-related issues in the future, don't hesitate to reach out to me for assistance.


You can also contact the National Consumer Helpline (1800-11-4000) for immediate assistance or visit their website at [www.nch.gov.in](http://www.nch.gov.in) for more information on consumer protection and redressal mechanisms.


Stay informed and empowered as a consumer!


Have a great day!'''
    first_line = extract_first_line(chat)
    const = {"old_chat":chat, "first_line":first_line}
    # Run the chat handler
    chat_handler(conversation_chain, user_chain, session_id, simulate_user=simulate_user, rag=True, const=const)
