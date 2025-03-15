# Contains hallucination promots and process chain with true/false and edit components
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


def process_chain(answer_chain, llm_engine_hf):
    # return answer_chain | RunnablePassthrough.assign(final_response=lambda l: l["answer"]) # If we need no processing
    # Processing Chain
    # process_system_prompt = '''Rewrite the following as if said by a 6 year old: {answer}'''
    # process_template = PromptTemplate(input_variables=["answer"], template=process_system_prompt, output_varables=["final_response"])
    
    # processing_chain =  process_template | llm_engine_hf  | StrOutputParser()

    # combined_chain = (answer_chain | RunnablePassthrough.assign(final_response=processing_chain))
    # process_system_prompt = '''You are an answer checker. The students were given a question where they had to extract information to answer some question. You do not know the question but you are given both the context and the answer by the students. Your job is for each sentence divide them into whether it is a fact based upon the context or it is fact out of context or if it is not a fact. Correctness of sentence does not matter, only its presence in the context does. Here is an example:
    process_system_prompt = """You are an examiner, and your job is to check whether the response contains information present in the context or if it includes external information. Your task is to repeat each sentence from the answer exactly as it is, but append one of the following labels:  

- (Inside Context) - The information is present in the provided context.  
- (Outside Context) - The information is NOT present in the context.  
- (General English) - The sentence is a general question, reasoning, or conversational filler not containing factual claims.  

Example:
Context:  
India has one of the largest cities in the world. Out of them, Mumbai has a population of 10 crore people and is considered the financial capital of India. Second-most, Chennai is also reputed to be an important port city.

Original Response:  
Delhi is the capital of India. Mumbai is the financial center of India. Where would you want to stay?

Analysis:  
Delhi is the capital of India. (Outside Context)  
Mumbai is the financial center of India. (Inside Context)  
Where would you want to stay? (General English)  

Now, classify each sentence in the following response according to these categories:  

Context:  
{context}  

Original Response:  
{answer}  

Analysis: 
"""

    fact_system_prompt = """You are a legal content reviewer. Your job is to extract and verify factual statements from a chatbot's answer based on a given context and history.  

Task:  
1. Extract factual statements from the answer. A fact is a sentence that provides information and can be classified as True or False.  
2. Evaluate the truthfulness of each fact based on the context:  
   - If the fact is true based upon the context, label it as [True].  
   - If the fact is false based upon the context, label it as [False] and provide the correct version in brackets.  
   - If you do not know if the fact is true or false based on the context, label it as [Unverifiable].
3. List non-factual statements separately (e.g., apologies, general conversational phrases, or questions). 
4. Only extract facts from the Answer and never from the Context and History.
5. Extract the exact sentences from the answer and do not paraphrase.

Example 1:

Context:  
The Consumer Protection Act allows consumers to file complaints against unfair trade practices. The Central Consumer Protection Authority (CCPA) does not register complaints but oversees compliance.  

History:
Chatbot: How can I help you with consumer disputes?
User: I'm trying to learn about consumer rights.

User: Answer:  
You can file a complaint with the CCPA for consumer disputes. The Consumer Protection Act provides legal remedies for consumers. Did you receive a response from the company?  

Chatbot:  

Facts:  
1. You can file a complaint with the CCPA for consumer disputes. [False: The CCPA does not register complaints.]  
2. The Consumer Protection Act provides legal remedies for consumers. [True]  

Other statements:  
1. Did you receive a response from the company?  

Example 2:


Context:
The Clean Water Act regulates discharges of pollutants into U.S. waters and quality standards for surface waters. The Environmental Protection Agency (EPA) oversees enforcement of the Clean Water Act, but state governments also have a role in its implementation.

History:
Chatbot: How can I help you with environmental regulations?
User: I'm trying to learn about water pollution laws.

User: Answer:
How can I help you with pollutiom law? Do you require a short description? Or do you need help with something else?

Chatbot:

Facts: No Fact exists

Other statements:
1. How can I help you with pollutiom law? 
2. Do you require a short description? 
3. Or do you need help with something else?


Now, perform the same fact extraction and classification for the following:  

Context:  
{context}  

History: 
{chat_history} 
"""
    fact_template = ChatPromptTemplate.from_messages([
        ("system", fact_system_prompt),
        ("user", "Now please extract the facts and classify them for the following Answer:\nAnswer: {answer}"),
    ])#PromptTemplate(input_variables=["context","answer"], template=fact_system_prompt)
    fact_chain =  fact_template | llm_engine_hf  | StrOutputParser()
    
    # process_system_prompt = '''You are a chatbot whose purpose is to help the user with consumer issues. You are given a rough draft made and a corresponding review done to point on the mistakes. Based on the False facts given in the review, modify the draft and rewrite that part using the context as the basis. In the parts where there is no errors which means no False fact present, then write those sentences as it is in draft. Also do not remove the thank you and sorry statements.
    process_system_prompt = """You are a Consumer Grievance Assistance Chatbot editor. Your task is to refine a chatbot's response by correcting incorrect factual statements while preserving all other content.  

Editing Guidelines:  
1. Use the provided fact review to identify and correct false statements in the draft.  
2. Retain all other statements exactly as they are, including but not limited to:  
   - Expressions of gratitude (e.g., "Thank you for reaching out.")  
   - Apologies (e.g., "I'm sorry to hear that.")  
   - Any questions present in the response.  
3. Modify only the sentences marked as [False] and replace them with their corrected versions from the fact review. Keep the [True] and [Unverifiable] statements without any changes.
4. Ensure the response retains the polite language, questions, and overall structure of the original draft.
5. Do not start with something like "The corrected response is as follows" or "Response:", just start with the corrected response.
6. The user should not know you are correcting the response. Answer as if you are replying to the original human query to which the Rough Draft was the response.


Example:  

Context:  
The Consumer Protection Act allows consumers to file complaints against unfair trade practices. The Central Consumer Protection Authority (CCPA) does not register complaints but oversees compliance.  

User: Rough Draft:  
You can file a complaint with the CCPA for consumer disputes. The Consumer Protection Act provides legal remedies for consumers. Did you receive a response from the company?  

Fact Review:  
1. You can file a complaint with the CCPA for consumer disputes. [False: The CCPA does not register complaints.]  
2. The Consumer Protection Act provides legal remedies for consumers. [True]  

Chatbot:  
The CCPA does not register complaints but oversees compliance. The Consumer Protection Act provides legal remedies for consumers. Did you receive a response from the company?  

Now, perform the same corrections based on the review provided.  

Context:  
{context}  
 """
    process_template = ChatPromptTemplate.from_messages([
        ("system", process_system_prompt),
        #MessagesPlaceholder("chat_history"),
        ("human", "Rough Draft: {answer}\n" "Fact Review: {review}"),
    ])
    processing_chain =  process_template | llm_engine_hf  | StrOutputParser()

    combined_chain = answer_chain | RunnablePassthrough.assign(review=fact_chain) | RunnablePassthrough.assign(final_response=processing_chain)

    # retrieval_docs = (lambda x: x["input"]) | answer_chain

    # retrieval_chain = (
    #     RunnablePassthrough.assign(
    #         context=retrieval_docs.with_config(run_name="retrieve_documents"),
    #     ).assign(answer=combine_docs_chain)
    # ).with_config(run_name="retrieval_chain")

    return combined_chain
    

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

    combined_chain =  process_chain(rag_chain, llm_engine_hf)

    return  combined_chain
