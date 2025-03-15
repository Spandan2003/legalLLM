# Contains hallucination promots and process chain
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
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
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
    process_system_prompt = """Rewrite the following response from a Consumer Grievance Assistance Chatbot to ensure it strictly aligns with the provided context. The response should:

1. Strictly adhere to the given context—remove any hallucinated or unsupported information.
2. Maintain the structure and wording of the original response, except for incorrect parts that need to be corrected.
3. It should begin as a direct response to the user—do not include "Response:" at the beginning.
4. You are only to check the legal and factual content for any inaccuracies. Do not modify any questions and the polite language used in the response like 
    - Thank you for sharing your concern with me. I'd be happy to help. 
    - Remember, you can also contact the National Consumer Helpline (NCH) at 1800-11-4000 or 1915 for immediate assistance.
    - Have a great day
    - Can I help you?

These are possible cases of hallucination:
1. Contradiction: Any part of the response that contradicts the given context or history.
2. Unsupported Information: Any facts or details that do not appear in the context and history both but are presented as factual in the response.
3. Fabricated Details: Any information such as contact numbers, addresses (detailed address also), email addresses, legal provisions, or company names that are not in the context nor in the history but present in the response.


Examples:

Example 1:
Input:
Context:
The Central Consumer Protection Authority (CCPA) is not a forum for consumer complaints. It cannot be used to register complaints but just governs the bodies.

Response:
You've purchased a product just 2 days ago, and it's still under warranty. Can you please tell me what led to the issue with the product? Was there a manufacturing defect, or perhaps it wasn't functioning as expected?

Output:
You've purchased a product just 2 days ago, and it's still under warranty. Can you please tell me what led to the issue with the product? Was there a manufacturing defect, or perhaps it wasn't functioning as expected?
Note: The response is correct and does not require any changes.

Example 2:
Input:
Context:
Consumer complaints against e-commerce platforms can be filed on the e-Daakhil portal. Complaints should include evidence such as invoices and communication records with the seller.

Response:
To file a complaint against an e-commerce platform, you can visit the e-Daakhil portal. Ensure that you include relevant evidence, such as invoices and communication records with the seller. You can also contact the customer care helpline 1800-4646-1200.

Output:
To file a complaint against an e-commerce platform, you can visit the e-Daakhil portal. Ensure that you include relevant evidence, such as invoices and communication records with the seller.
Note: The response contains a number which is not present in the context.


Input:
Context:
{context}

Response:
{answer}

Output:
"""
    process_template = PromptTemplate(input_variables=["answer"], template=process_system_prompt, output_varables=["final_response"])
    
    processing_chain = process_template | llm_engine_hf | StrOutputParser() | RunnableLambda(lambda l: l.split("Note:")[0].strip() if "Note:" in l else l)

    combined_chain = (answer_chain | RunnablePassthrough.assign(final_response=processing_chain))

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

    combined_chain =  process_chain(rag_chain, llm_engine_hf)

    return combined_chain
