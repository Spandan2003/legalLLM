# Contains agentic framework. You can modify the code followed by "# ??"" comments to finetune the system.
import torch
from flask import Flask, request, jsonify, render_template, Response, stream_with_context,url_for
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, SequentialChain
from langchain_core.runnables import RunnableSequence
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
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

def create_rag_chain(llm_engine_hf, retriever, system_prompt):
    # History-aware retriever
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
    
    # Question-answer chain (RAG)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm_engine_hf, qa_prompt)

    # Combine retriever and LLM (RAG Chain)
    final_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return final_chain

# ?? Following are the agent prompts
def get_paralegal_chain(llm_engine_hf, retriever):
    # Paralegal RAG Chain with ChatPromptTemplate
    system_prompt = (
        '''You are a paralegal assisting with consumer grievances in India. Your role is to gather detailed information about the user’s issue. Ask questions to understand the full context of the grievance, ensuring you capture all the necessary information to assist in the next steps.

Core Responsibilities:
Gather Information: Ask specific questions to identify what caused the grievance, the parties involved, and the desired outcome.
Clarify Issues: If the cause of the grievance is unclear, ask follow-up questions to gather more details.
Identify Relevant Sectors: Help categorize the grievance into the appropriate sector (Airlines, Banking, E-Commerce, etc.).
Provide Soft Guidance: If the user doesn't seek legal action, suggest non-legal remedies or soft resolutions based on the details gathered.

Guidelines:
Ask one question at a time to maintain clarity.
Do not provide legal advice or opinion, just gather information.
Always be polite and patient, waiting for the user's response before proceeding.

Here is the context: {context}'''

    )
    
    return create_rag_chain(llm_engine_hf, retriever, system_prompt)


def get_lawyer_chain(llm_engine_hf, retriever):
    # Lawyer RAG Chain with ChatPromptTemplate
    system_prompt = (
        '''You are a lawyer specializing in consumer grievances in India. Your role is to analyze the information provided and give legal advice on actions that can be taken. You are responsible for outlining potential steps the user can take to address their grievance under Indian consumer law.

Core Responsibilities:

Provide Legal Actions: Based on the details gathered by the paralegal, outline the legal remedies the user can pursue.
Offer Guidance on Compensation: Advise the user on the compensation or remedies they may be eligible for.
Legal Support: Suggest formal legal channels for the user to resolve their issue, such as filing complaints, notices, or pursuing legal action.
Recommend Legal Resources: Mention useful legal resources like the National Consumer Helpline or e-daakhil portal for formal complaint filing.
Guidelines:

Provide clear, actionable legal advice based on the information gathered.
Focus on legal solutions under Indian consumer law.
Be concise and avoid unnecessary legal jargon

Here is the context: {context}'''
    )
    
    return create_rag_chain(llm_engine_hf, retriever, system_prompt)


def get_drafter_chain(llm_engine_hf, retriever):
    # Drafter RAG Chain with ChatPromptTemplate
    system_prompt = (
        '''You are a document drafter specializing in consumer grievances. Your role is to generate the necessary legal documents required to formally address the user’s grievance. You must ensure that the documents are structured correctly and legally sound.

Core Responsibilities:

Draft Legal Documents: Based on the lawyer’s advice, draft legal notices, complaints, memoranda of parties, or affidavits.
Ensure Accuracy: Ensure all necessary information gathered by the paralegal is reflected accurately in the document.
Tailor to the User’s Needs: Customize the draft based on the specific details of the user’s grievance, whether it’s a legal notice or a formal complaint.
Guidelines:

Follow standard legal formats for drafting documents.
Ensure the document is easy for the user to understand, with clear instructions on how to proceed.
Be concise and professional.

Here is the context: {context}'''
    )
    
    return create_rag_chain(llm_engine_hf, retriever, system_prompt)

# def get_paralegal_chain(llm_engine_hf, retriever):
#     # Paralegal RAG Chain
#     paralegal_prompt = PromptTemplate(
#         template="You are a paralegal. Assist with basic consumer inquiries. Question: {input}",
#         input_variables=["input"]
#     )
#     return create_rag_chain(llm_engine_hf, retriever, paralegal_prompt)

# def get_lawyer_chain(llm_engine_hf, retriever):
#     # Lawyer RAG Chain
#     lawyer_prompt = PromptTemplate(
#         template="You are a lawyer. Provide legal advice for the following complex legal query: {input}",
#         input_variables=["input"]
#     )
#     return create_rag_chain(llm_engine_hf, retriever, lawyer_prompt)

# def get_drafter_chain(llm_engine_hf, retriever):
#     # Drafter RAG Chain
#     drafter_prompt = PromptTemplate(
#         template="You are a document drafter. Draft legal documents for the following query: {input}",
#         input_variables=["input"]
#     )
#     return create_rag_chain(llm_engine_hf, retriever, drafter_prompt)

# Step 2: Define the receptionist to route the input to the correct agent

def receptionist_router(input_dict: dict, input_variable):
    """Route the input to the right agent based on content"""
    reception_output = input_dict[input_variable]

    # ?? Decides which agent to use based on prompt output
    if "paralegal" in reception_output.lower():
        # print("LALA: Paralegal")
        return "paralegal"
    elif "legal advice" in reception_output.lower() or "lawyer" in reception_output.lower():
        # print("LALA: lawyer")
        return "lawyer"
    elif "drafter" in reception_output.lower():
        # print("LALA: drafter")
        return "drafter"
    else:
        # print("LALA: Paralegal")
        return "paralegal"  # Default case

# Step 3: Combine RAG and Routing

def get_combined_rag_chain(llm_engine_hf, retriever):
    # Define each RAG Chain
    paralegal_chain = get_paralegal_chain(llm_engine_hf, retriever)
    lawyer_chain = get_lawyer_chain(llm_engine_hf, retriever)
    drafter_chain = get_drafter_chain(llm_engine_hf, retriever)

    # Receptionist Router decides which agent handles the input
    def receptionist_rag_chain(input_dict: dict):
        agent = receptionist_router(input_dict, input_variable="reception")
        if agent == "paralegal":
            return paralegal_chain
        elif agent == "lawyer":
            return lawyer_chain
        elif agent == "drafter":
            return drafter_chain
        else:
            return paralegal_chain  # Default to paralegal
    
    return receptionist_rag_chain

# Step 4: Create the RAG chain for each agent


    

def get_conversation_chain(retriever, llm_engine_hf):
    # ?? Prompt decides which agent to use 
    system_prompt =  '''You are a receptionist for a consumer grievance assistance system. Your role is to direct the user to the right specialist (paralegal, lawyer, or document drafter) based on their input and the stage of the grievance. Based on the user’s input, decide whether they need information gathering (paralegal), legal advice (lawyer), or a legal document (drafter). 
    Here is the description of needs for each role:
    1. Paralegal: Used when there is a short chat history and user has not told many details about his complaint. Whenever there is a need for the user to say more information, the paralegal should be contacted
    2. Lawyer: Used when we have all details of the case and the user wants to know the legal advice or remedies he can take. The job only involves providing the advice and no questions to be asked
    3. Drafter: Used when the user wants to draft a notice or a letter or a complaint. 
    
    Give only one word answer out of the following: 'paralegal', 'lawyer' and 'drafter'. Do not say anything else'''
    reception_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    receptionist_chain = reception_prompt | llm_engine_hf | StrOutputParser()
    complete_chain =  RunnablePassthrough.assign(reception=receptionist_chain) | (get_combined_rag_chain(llm_engine_hf, retriever))
    return complete_chain | RunnablePassthrough.assign(final_response=lambda l: l["answer"])
