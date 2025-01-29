import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
import os
import uuid
import json
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used is ", device)

# Specify the folder containing your PDF files
PDF_FOLDER = '/data/nlp/spandan/code/legalLLM/chatbot/general_chatbot/dataset/rag'

# Global variables
vectorstore = None
consumer_conversation_chain = None
general_conversation_chain = None

# Statefully manage chat history
consumer_store = {}
general_store = {}


def get_consumer_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in consumer_store:
        consumer_store[session_id] = ChatMessageHistory()
        # Add the initial AI message to the chat history
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        consumer_store[session_id].add_message(AIMessage(content=initial_message))
    return consumer_store[session_id]

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
    embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs={"device": device})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_llm():
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        task="text-generation",
        device=0,
        callbacks=callbacks,
        pipeline_kwargs=dict(
            return_full_text=False,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.5,
        ),
        
    )
    llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id
    llm.pipeline.model.config.pad_token_id = llm.pipeline.model.config.eos_token_id
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

    # Hook to capture reformulated question output
    def reformulate_question(input, history):
        reformulated_question = history_aware_retriever.invoke({
            "input": input, 
            "chat_history": history
        })
        print("Reformulated Question:", reformulated_question)
        return reformulated_question

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
Do not let the user know you answered the question using the context.
\n\n
{context}
'''
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm_engine_hf, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain, reformulate_question

def initialize():
    global vectorstore, consumer_conversation_chain
    print("Initializing...")
    
    # Process documents
    raw_text = get_pdf_text(PDF_FOLDER)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever()
    print("Documents processed successfully")
    
    # Initialize LLM
    llm_engine_hf = get_llm()
    
    # Initialize conversation chain
    consumer_conversation_chain = get_conversation_chain(retriever, llm_engine_hf)
    print("Conversation chain initialized")

def chat():
    session_id = str(uuid.uuid4())
    history = get_consumer_session_history(session_id)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        #print(datetime.now())
        #print("user:", user_input)

        conversational_rag_chain = RunnableWithMessageHistory(
            consumer_conversation_chain,
            get_consumer_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        config = {"configurable": {"session_id": session_id}}
        response = conversational_rag_chain.invoke({"input": user_input}, config)
        for i in range(len(response['context'])):
            print("Retrieval " + str(i) + ":", response['context'][i].page_content.replace('\n', ' '))
        print("Chatbot:", response["answer"])
        print(response)

if __name__ == '__main__':
    initialize()
    chat()
