# Prompt has been modified to ensure that model only uses context from the content and history to give output.
# Second modification to ensure forum issue is resolved
import torch
from flask import Flask, request, jsonify, render_template, Response, stream_with_context,url_for
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
# from langchain_community.llms import HuggingFacePipeline, Chat
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
import os
import uuid
import json
from flask_cors import CORS 
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage
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
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    callbacks = [StreamingStdOutCallbackHandler()]
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_id,
    #     task="text-generation",
    #     device=0,
    #     callbacks = callbacks,
    #     pipeline_kwargs=dict(
    #         return_full_text=False,
    #         max_new_tokens=1024,
    #         do_sample=True,
    #         temperature=0.5,
    #     ),
    # )
    # llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id
    # llm_engine_hf = ChatHuggingFace(llm=llm)
    llm_engine_hf = ChatOpenAI(
        # model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        openai_api_key="EMPTY",
        openai_api_base="http://172.17.0.1:8080/v1/",
        max_tokens=1024,
        temperature=0.5,
        )
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

def get_general_chain(llm_engine_hf):
    system_prompt = "Hello! I'm your General Knowledge Assistant. How can I help you today?"
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = qa_prompt | llm_engine_hf

    return question_answer_chain

def initialize_app():
    global vectorstore, consumer_conversation_chain, general_conversation_chain
    print("Initializing application...")
    
    # Process documents
    raw_text = get_pdf_text(PDF_FOLDER)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever()
    print("Documents processed successfully")
    
    # Intialize llm 
    llm_engine_hf = get_llm()
    # Initialize conversation chain
    consumer_conversation_chain = get_conversation_chain(retriever,llm_engine_hf)
    general_conversation_chain = get_general_chain(llm_engine_hf)

    print("Conversation chain initialized")

app = Flask(__name__, static_url_path='/consumer_chatbot/static')
CORS(app)
app.secret_key = os.urandom(24)
load_dotenv()

with app.app_context():
    initialize_app()

@app.route(f'/{BASE_URL}')
def index():
    return render_template('index.html', BASE_URL=BASE_URL)
@app.route(f'{BASE_URL}/general')
def general_index():
    return render_template('general_index.html', BASE_URL=f'{BASE_URL}/general')

@app.route(f'{BASE_URL}/get_session_id', methods=['GET'])
@app.route(f'{BASE_URL}/general/get_session_id', methods=['GET'])
def get_session_id():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})

@app.route(f'/{BASE_URL}/initial_message', methods=['GET'])
def initial_message():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    history = get_consumer_session_history(session_id)
    initial_ai_message = history.messages[0].content if history.messages else "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
    return jsonify({"message": initial_ai_message})

@app.route(f'{BASE_URL}/general/initial_message', methods=['GET'])
def general_initial_message():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    history = get_general_session_history(session_id)
    initial_ai_message = history.messages[0].content if history.messages else "Hello! I'm your General Knowledge Assistant. How can I help you today?"
    return jsonify({"message": initial_ai_message})

@app.route(f'{BASE_URL}/chat', methods=['POST', 'GET'])
def consumer_chat():
    return chat_handler(consumer_conversation_chain, get_consumer_session_history, rag=True)

@app.route(f'{BASE_URL}/general/chat', methods=['POST', 'GET'])
def general_chat():
    return chat_handler(general_conversation_chain, get_general_session_history, rag=False)

def chat_handler(conversation_chain, get_session_history_func, rag):
    if request.method == 'POST':
        data = request.json
    else:  # GET
        data = request.args
    user_input = data.get('message')
    session_id = data.get('session_id')
    print(datetime.now())
    print("history", get_session_history_func(session_id))
    print("user:", user_input)
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    if rag:
        conversational_rag_chain = RunnableWithMessageHistory(
            conversation_chain,
            get_session_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        ) 
    else:
        conversational_rag_chain = RunnableWithMessageHistory(
            conversation_chain,
            get_session_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
            # output_messages_key="answer",
        ) 
    parser = StrOutputParser()
    chain = conversational_rag_chain 
    config = {"configurable": {"session_id": session_id}}
    response = chain.invoke({"input": user_input}, config)
    if rag:
        print("response", response)
        formatted_response = response['answer'].replace('\n', '<br>')
    else:
        print("response", response)
        formatted_response = response.content.replace('\n', '<br>')
    return jsonify({"response": formatted_response}), 200

@app.route(f'{BASE_URL}/get_chat_history', methods=['GET'])
def get_consumer_chat_history():
    return get_chat_history(get_consumer_session_history)

@app.route(f'{BASE_URL}/general/get_chat_history', methods=['GET'])
def get_general_chat_history():
    return get_chat_history(get_general_session_history)

def get_chat_history(get_session_history_func):
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    history = get_session_history_func(session_id)
    print("history",history)
    chat_history = [
        {"role": "AI" if isinstance(msg, AIMessage) else "Human", "content": msg.content}
        for msg in history.messages
    ]
    return jsonify({"chat_history": chat_history})

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=False,port=60000)