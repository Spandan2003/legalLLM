# Contains hallucination prompts and hallucination detector to finalize the response
import torch
from flask import Flask, request, jsonify, render_template, Response, stream_with_context,url_for
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
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
    # process_system_prompt = '''Rewrite the following as if said by a 6 year old: {answer}'''
    # process_template = PromptTemplate(input_variables=["answer"], template=process_system_prompt, output_varables=["final_response"])
    
    # processing_chain =  process_template | llm_engine_hf  | StrOutputParser()

    # combined_chain = (answer_chain | RunnablePassthrough.assign(final_response=processing_chain))
    # process_system_prompt = '''You are an answer checker. The students were given a question where they had to extract information to answer some question. You do not know the question but you are given both the context and the answer by the students. Your job is for each sentence divide them into whether it is a fact based upon the context or it is fact out of context or if it is not a fact. Correctness of sentence does not matter, only its presence in the context does. Here is an example:
    detector_system_prompt = ("You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot which is to give correct and flawless legal advice to the user. For this you are provided the context (the textual knowledge related to the query), History (the previous chat between the user and the chatbot), the Query (question or statement by the user on which the response is based) and the Response (by the chatbot)\n"
                "Your goal is to review the response and point out the inconsistencies or falsehood in the chatbot. This is a list of the possible faults\n"
                "1. Any response which contradicts the context or any part of the context\n"
               # "2. Any response that has information or facts which do not exist in the context and history\n"
                "3. Any response that has information like contact number or address or so on of a company, organization or government body that has not been mentioned in the context or in the history\n"
                "Your goal is to review the response and find out whether there are inconsistencies or falsehood in the chatbot.\n"
                "Ensure that if there are inconsistencies then they should be written in a concise manner and only the major inconsistencies should be pointed out"
                "Only state the assumptions which are based upon false statements. If any assumption is based upon logic, do not mention it.\n"
                "Only point out the inconsistencies which are regarding facts. Do not state the inconsistencies about statements obtained by reasoning.\n"
                "Ensure that the response is short, concise, without repetition and ends with an `<|endoftext|>` token"
                "\nExample 1:\n"
                "Context:\n"
                "The Employment Rights Act 1996 provides employees with certain rights, such as the right to request flexible working, protection against unfair dismissal, and statutory redundancy pay. It does not mention any provisions about maternity benefits or sick pay directly.\n\n"
                "History:\n"
                "User: What rights do employees have under the Employment Rights Act 1996?\n"
                "Chatbot: Employees have the right to flexible working, protection against unfair dismissal, and statutory redundancy pay.\n"
                "User: Does it include maternity benefits?\n"
                "Chatbot: Yes, the Act includes maternity benefits for employees.\n\n"
                "Query:\n"
                "Can you provide more details about maternity benefits under this Act?\n\n"
                "Response:\n"
                "Maternity benefits under the Employment Rights Act 1996 include 26 weeks of maternity leave and statutory maternity pay for eligible employees.\n\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The response mentions maternity benefits under the Employment Rights Act 1996, but the context explicitly states that the Act does not mention provisions about maternity benefits.\n"
                "2. The details about \"26 weeks of maternity leave\" and \"statutory maternity pay\" are fabricated and not supported by the context.\n<|endoftext|>"

                "\nExample 2:\n"
                "Context:\n"
                "Under the Consumer Rights Act 2015, consumers are entitled to a refund, repair, or replacement for goods that are faulty or not as described. The Act does not include any references to warranty periods being extended for defective goods.\n\n"
                "History:\n"
                "User: Can I get an extended warranty if my product is faulty under the Consumer Rights Act 2015?\n"
                "Chatbot: Yes, the Act allows for an extended warranty period for defective goods.\n"
                "User: What kind of extended warranty can I expect?\n"
                "Chatbot: The warranty duration depends on the product type and its condition.\n\n"
                "Query:\n"
                "How long is the extended warranty period?\n\n"
                "Response:\n"
                "The Consumer Rights Act 2015 provides a 12-month extended warranty for defective goods.\n\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The response falsely claims that the Act provides for an extended warranty period, which is explicitly not mentioned in the context.\n"
                "2. The \"12-month extended warranty\" detail is fabricated and unsupported by the context.\n<|endoftext|>"

                "\nExample 3:\n"
                "Context:\n"
                "Section 21 of the Housing Act 1988 allows landlords to evict tenants without providing a reason, as long as the proper notice period is given. The notice period is currently two months.\n\n"
                "History:\n"
                "User: Can a landlord evict a tenant without giving a reason under Section 21?\n"
                "Chatbot: Yes, under Section 21, landlords can evict tenants without giving a reason.\n"
                "User: What is the notice period for this kind of eviction?\n"
                "Chatbot: I believe it’s one month, but I’ll confirm.\n\n"
                "Query:\n"
                "Can you confirm if one month is the required notice period?\n\n"
                "Response:\n"
                "Yes, the notice period under Section 21 is one month.\n\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The response incorrectly states that the notice period under Section 21 is one month, while the context explicitly states it is two months.\n<|endoftext|>"

                "\nExample 4:\n"
                "Context:\n"
                "The Data Protection Act 2018 governs how personal data should be processed and stored. It includes principles such as lawfulness, fairness, transparency, and data minimization. The Act does not mention specific fines for breaches.\n\n"
                "History:\n"
                "User: What is the main purpose of the Data Protection Act 2018?\n"
                "Chatbot: The Act ensures personal data is processed lawfully and transparently.\n"
                "User: Are there any fines for breaches mentioned in the Act?\n"
                "Chatbot: Yes, there are specific fines.\n\n"
                "Query:\n"
                "What are the fines for breaches under this Act?\n\n"
                "Response:\n"
                "The Data Protection Act 2018 specifies fines of up to £20 million for breaches.\n\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The response falsely claims that the Data Protection Act 2018 specifies fines of up to £20 million, while the context explicitly states that the Act does not mention specific fines.\n<|endoftext|>"

                "\nExample 5:\n"
                "Context:\n"
                "The Road Traffic Act 1988 mandates that all drivers must have third-party liability insurance. It does not specify penalties for not having insurance.\n\n"
                "History:\n"
                "User: Does the Road Traffic Act 1988 require insurance?\n"
                "Chatbot: Yes, all drivers must have third-party liability insurance.\n"
                "User: What happens if someone doesn’t have insurance?\n"
                "Chatbot: I’m not sure about the exact penalties.\n\n"
                "Query:\n"
                "Can you confirm the penalties under this Act?\n\n"
                "Response:\n"
                "Under the Road Traffic Act 1988, driving without insurance results in a penalty of £300 and six points on the driver’s license.\n\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The response incorrectly states that the penalty is mentioned in the Act, whereas the context explicitly states that the Act does not specify penalties.\n<|endoftext|>"

"Example 6:\n"
"Context:\n"
"The Health and Safety at Work Act 1974 ensures the safety, health, and welfare of employees at work. Employers must conduct regular risk assessments and provide necessary safety equipment. The Act does not mention mental health initiatives or stress management requirements explicitly.\n\n"
"History:\n"
"AI: Hello! How can I assist with workplace safety issues?\n"
"User: My employer doesn't provide safety equipment.\n"
"AI: Employers are required to provide necessary safety equipment under the Health and Safety at Work Act 1974. Can you share more about the equipment missing or the risks involved?\n\n"
"Query:\n"
"The equipment is missing entirely.\n\n"
"Response:\n"
"If your employer has not provided any safety equipment, they are in violation of the Health and Safety at Work Act 1974. You can report this to the Health and Safety Executive (HSE). Additionally, the Act mandates that employers must address workplace stress and mental health issues, which can also be reported.\n\n"
"Inconsistencies Present: Yes\n"
"Inconsistencies:\n"
"1. The response claims that the Act mandates addressing workplace stress and mental health issues, which is not supported by the combination of context and history.\n<|endoftext|>\n\n"
"Example 7:\n"
"Context:\n"
"The General Data Protection Regulation (GDPR) provides individuals with the right to access, correct, and delete their personal data. It also mandates that data breaches must be reported to the relevant authority within 72 hours. The GDPR does not specify penalties for breaches.\n\n"
"History:\n"
"AI: Hi! How can I assist with data protection concerns?\n"
"User: I suspect my personal data was leaked by a company.\n"
"AI: I’m sorry to hear that. Have you reported the issue to the company or checked their data breach policy?\n\n"
"Query:\n"
"I haven’t reported it yet.\n\n"
"Response:\n"
"If you suspect a data breach, you should notify the company and request details under GDPR regulations. Additionally, the GDPR allows individuals to claim compensation for data breaches directly from the company.\n\n"
"Inconsistencies Present: Yes\n"
"Inconsistencies:\n"
"1. The response incorrectly claims that GDPR allows individuals to claim compensation for breaches directly, which is not supported by the combination of context and history.\n<|endoftext|>\n\n"
"Example 8:\n"
"Context:\n"
"The Sale of Goods Act 1979 gives consumers the right to reject goods that are not of satisfactory quality, not fit for purpose, or not as described. Consumers must act within 30 days of purchase to claim a refund. The Act does not cover services or digital products.\n\n"
"History:\n"
"AI: How can I help with your recent purchase?\n"
"User: I bought a defective laptop.\n"
"AI: Under the Sale of Goods Act 1979, you’re entitled to a remedy if the laptop is defective. Did you report this issue within 30 days of purchase?\n\n"
"Query:\n"
"Yes, I reported it within the time limit.\n\n"
"Response:\n"
"If you reported the defect within 30 days, you are entitled to a full refund under the Sale of Goods Act 1979. If the seller refuses, you can escalate the matter to trading standards. Would you like assistance with drafting a complaint or contacting trading standards?\n\n"
"Inconsistencies Present: No\n<|endoftext|>\n\n"
"Example 9:\n"
"Context:\n"
"Section 21 of the Housing Act 1988 allows landlords to evict tenants without giving a reason, provided they give a notice period of at least two months. The Act does not mention any rights to compensation for tenants upon eviction.\n\n"
"History:\n"
"AI: Hi! How can I help with your housing issue today?\n"
"User: My landlord has served me a Section 21 notice.\n"
"AI: I see. Was the notice served with at least a two-month period as required under the Housing Act 1988?\n\n"
"Query:\n"
"No, it was less than two months.\n\n"
"Response:\n"
"If the notice period is less than two months, it is not valid under Section 21 of the Housing Act 1988. Additionally, you may be entitled to compensation if the landlord has breached any of the regulations under this Act.\n\n"
"Inconsistencies Present: Yes\n"
"Inconsistencies:\n"
"1. The response incorrectly states that tenants may be entitled to compensation under the Housing Act 1988, which is not supported by the combination of context and history.\n<|endoftext|>\n\n"
"Example 10:\n"
"Context:\n"
"The Equality Act 2010 protects individuals from discrimination in employment, education, and access to services. It prohibits unfair treatment based on characteristics such as age, disability, gender, and race. The Act does not include provisions for harassment on social media.\n\n"
"History:\n"
"AI: How can I assist you with discrimination concerns?\n"
"User: I faced discrimination due to my disability at work.\n"
"AI: I’m sorry to hear that. Did you file a formal complaint with your employer?\n\n"
"Query:\n"
"Yes, I filed a complaint.\n\n"
"Response:\n"
"Filing a formal complaint is a great first step. Under the Equality Act 2010, your employer is required to investigate and address the issue. If no action is taken, you can escalate the matter to an employment tribunal. Would you like guidance on how to proceed with the tribunal?\n\n"
"Inconsistencies Present: No\n<|endoftext|>\n\n"
"Example 11:\n"
"Context:\n"
"The Road Traffic Act 1988 mandates that all drivers must have third-party liability insurance. It does not specify penalties for failing to have insurance, but uninsured driving is illegal and subject to separate legal penalties.\n\n"
"History:\n"
"AI: Hello! How can I help with road traffic issues?\n"
"User: I was caught driving without insurance.\n"
"AI: Driving without insurance is illegal. Were you informed of any penalties by the authorities?\n\n"
"Query:\n"
"No, they didn’t mention any penalties.\n\n"
"Response:\n"
"Under the Road Traffic Act 1988, uninsured driving is prohibited. While the Act doesn’t specify penalties, common consequences include fines or points on your license under other laws. Would you like help contacting legal aid or understanding the process further?\n\n"
"Inconsistencies Present: No\n<|endoftext|>"



                "Context:\n {context}\nContact the National Consumer Helpline (1800-11-4000) or UMANG App for immediate assistance.\n"
                "History:\n"
                )

    detector_template = ChatPromptTemplate.from_messages([
        ("system", detector_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Query:\n {input}\n"
                "Response:\n {answer}\n"),
    ])
    detector_chain =  detector_template | llm_engine_hf  | StrOutputParser()

    editor_system_prompt = ( 
                    "You are an expert legal assistant whose job is to provide accurate and professional responses to user queries based on the given context. You are provided with the following inputs:\n"
"1. Response: The original response provided by the chatbot.\n"
"2. Context: The relevant legal or factual knowledge related to the query.\n"
"3. Inconsistencies: The inconsistencies or falsehoods identified in the original response.\n\n"
"Your task is to rewrite the response to:\n"
"- Correct any inconsistencies pointed out.\n"
"- Ensure the revised response aligns with the provided context.\n"
"- Avoid making any changes other than addressing the listed inconsistencies.\n"
"- Ensure that the content of the response does not change. If it has a question then keep the question while making it factually correct.\n"


"Format:\n"
"Revised Response: [Your accurate and corrected response here]\n\n"
"Examples:\n\n"
"Example 1:\n"
"History:\n"
"AI: What can I assist you with today?\n"
"User: I need help with a data breach issue.\n"
"AI: The Data Protection Act 2018 covers data breaches. Did you lose money or suffer distress because of the breach?\n\n"
"Query:\n"
"Yes, I suffered financial loss.\n\n"
"Response:\n"
"If you've suffered financial loss due to a data breach, you may be entitled to compensation under the Data Protection Act 2018. The organization responsible must pay a fixed penalty of £5,000 for each affected individual. Have you already reported the breach to the organization, or would you like help with filing a complaint?\n\n"
"Context:\n"
"The Data Protection Act 2018 allows individuals to seek compensation for financial or emotional damages caused by a data breach. However, the Act does not mandate a fixed penalty of £5,000 per individual. Compensation amounts are determined on a case-by-case basis.\n\n"
"Inconsistencies Present: Yes\n"
"Inconsistencies:\n"
"1. The claim about a fixed penalty of £5,000 is unsupported by the context.\n\n"
"Revised Response:\n"
"If you've suffered financial loss due to a data breach, you may be entitled to compensation under the Data Protection Act 2018. Compensation amounts depend on the circumstances of the breach. Have you already reported the breach to the organization, or would you like help with filing a complaint?\n\n"
"<|endoftext|>\n\n"

"Example 2:\n"
"History:\n"
"AI: How can I assist you regarding your employment rights?\n"
"User: I believe I’ve been unfairly dismissed.\n"
"AI: Under the Employment Rights Act 1996, you can challenge unfair dismissal. Did you work for the employer for at least two years before dismissal?\n\n"
"Query:\n"
"Yes, I worked there for three years.\n\n"
"Response:\n"
"If you worked for the employer for at least two years, you are eligible to challenge unfair dismissal under the Employment Rights Act 1996. Your employer must pay you compensation equal to six months’ salary immediately. Have you received a formal dismissal letter, and would you like guidance on filing a claim with an employment tribunal?\n\n"
"Context:\n"
"The Employment Rights Act 1996 allows employees to challenge unfair dismissal if they have worked for the employer for at least two years. Compensation for unfair dismissal is determined by the tribunal and depends on the employee's circumstances. The Act does not mandate immediate compensation equal to six months’ salary.\n\n"
"Inconsistencies Present: Yes\n"
"Inconsistencies:\n"
"1. The claim about immediate compensation of six months’ salary is unsupported by the context.\n\n"
"Revised Response:\n"
"If you worked for the employer for at least two years, you are eligible to challenge unfair dismissal under the Employment Rights Act 1996. Compensation is determined by the tribunal based on your circumstances. Have you received a formal dismissal letter, and would you like guidance on filing a claim with an employment tribunal?\n\n"
"<|endoftext|>\n\n"

"Example 3:\n"
"History:\n"
"AI: Do you need help with a consumer issue?\n"
"User: Yes, I purchased a faulty washing machine.\n"
"AI: Under the Consumer Rights Act 2015, you’re entitled to a remedy if the product is faulty. Did the defect occur within the first six months of purchase?\n\n"
"Query:\n"
"Yes, it stopped working two months after I bought it.\n\n"
"Response:\n"
"If the washing machine stopped working within the first six months, you’re entitled to a full refund, and the retailer must compensate you for inconvenience caused. Would you like assistance in contacting the retailer or drafting a formal complaint?\n\n"
"Context:\n"
"The Consumer Rights Act 2015 allows consumers to request a refund, repair, or replacement of faulty goods within six months of purchase. However, compensation for inconvenience caused is not mandated under the Act.\n\n"
"Inconsistencies Present: Yes\n"
"Inconsistencies:\n"
"1. The claim that the retailer must compensate for inconvenience caused is unsupported by the context.\n\n"
"Revised Response:\n"
"If the washing machine stopped working within the first six months, you’re entitled to a refund, repair, or replacement under the Consumer Rights Act 2015. Would you like assistance in contacting the retailer or drafting a formal complaint?\n\n"
"<|endoftext|>\n\n"

"Example 4:\n"
"History:\n"
"AI: What brings you here today?\n"
"User: My employer isn’t paying me equally compared to my colleagues.\n"
"AI: The Equality Act 2010 ensures equal pay for equal work. Are your colleagues performing the same job or a similar role?\n\n"
"Query:\n"
"Yes, they have the same role and responsibilities.\n\n"
"Response:\n"
"If your colleagues with the same role and responsibilities are being paid more, this is a direct violation of the Equality Act 2010. Your employer is legally required to double your salary as compensation. Do you have evidence of the pay disparity, and would you like guidance on raising the issue with your employer?\n\n"
"Context:\n"
"The Equality Act 2010 ensures equal pay for equal work and allows employees to challenge pay disparities. Compensation for proven pay discrimination is determined by an employment tribunal. The Act does not mandate doubling an employee’s salary as compensation.\n\n"
"Inconsistencies Present: Yes\n"
"Inconsistencies:\n"
"1. The claim that the employer must double the employee's salary is unsupported by the context.\n\n"
"Revised Response:\n"
"If your colleagues with the same role and responsibilities are being paid more, this may violate the Equality Act 2010. Compensation for proven pay disparities is determined by an employment tribunal. Do you have evidence of the pay disparity, and would you like guidance on raising the issue with your employer?\n\n"
"<|endoftext|>\n\n"

"Example 5:\n"
"History:\n"
"AI: How can I help you with your legal query?\n"
"User: I want to sue my landlord for unsafe living conditions.\n"
"AI: Under the Landlord and Tenant Act 1985, landlords must maintain safe living conditions. Have you already notified your landlord about the issues?\n\n"
"Query:\n"
"Yes, but they ignored my complaints.\n\n"
"Response:\n"
"If your landlord ignored your complaints, you can sue them for negligence under the Landlord and Tenant Act 1985. They are required to pay compensation for every month the property remained unsafe. Do you have documentation of your complaints and the unsafe conditions? Would you like help with drafting a formal legal notice or finding legal representation?\n\n"
"Context:\n"
"The Landlord and Tenant Act 1985 requires landlords to maintain safe living conditions. Tenants can report landlords to the local council or take legal action, but the Act does not mandate monthly compensation for unsafe conditions.\n\n"
"Inconsistencies Present: Yes\n"
"Inconsistencies:\n"
"1. The claim about compensation for every month the property remained unsafe is unsupported by the context.\n\n"
"Revised Response:\n"
"If your landlord ignored your complaints, you can take legal action under the Landlord and Tenant Act 1985. Do you have documentation of your complaints and the unsafe conditions? Would you like help with drafting a formal legal notice or finding legal representation?\n\n"
"<|endoftext|>\n\n")

    editor_template = ChatPromptTemplate.from_messages([
        ("system", editor_system_prompt),
        ("human", "Response: \n{answer}\n"
                    "Context:\n {context}\nContact the National Consumer Helpline (1800-11-4000) or UMANG App for immediate assistance.\n" 
                    "{inconsistencies}\nRevised Response: ")
    ])
    editor_chain =  editor_template | llm_engine_hf  | StrOutputParser()
    editor_chain_mod = RunnableLambda(
                lambda inputs: inputs["answer"]  # Directly return response if no inconsistencies
                if inputs["inconsistencies"].startswith("Inconsistencies Present: No")
                else editor_chain.invoke(inputs).removeprefix("Revised Response: \n").removeprefix("Revised Response:\n")  )

    combined_chain = answer_chain | RunnablePassthrough.assign(inconsistencies=detector_chain) | RunnablePassthrough.assign(final_response=editor_chain_mod)

    # retrieval_docs = (lambda x: x["input"]) | answer_chain

    # retrieval_chain = (
    #     RunnablePassthrough.assign(
    #         context=retrieval_docs.with_config(run_name="retrieve_documents"),
    #     ).assign(answer=combine_docs_chain)
    # ).with_config(run_name="retrieval_chain")

    return combined_chain
    

def get_conversation_chain(retriever, llm_engine_hf):

    contextualize_q_system_prompt = (
            "Given the chat history and the latest user statement (Query) which might refer to the chat history, formulate a standalone statement which can be understood without the chat history. Do NOT answer the Query, just reformulate the Query only if needed and otherwise return it as it is. The query can either be a question or a answer so ensure you reformulate it properly in both cases\n\n"
            "For example:\n"
            "Example 1\n"
            "Chat History:\n"
            "Human: What is Task Decomposition?\n"
            "AI: Task Decomposition is the process of breaking down a complex task into smaller and simpler steps. This is achieved through a technique called Chain of Thought (CoT), which instructs the model to \"think step by step\" and utilize more test-time computation to transform big tasks into multiple manageable tasks.\n"
            "Human: Can you provide an example of Chain of Thought reasoning?\n"
            "AI: Sure! Let’s say the task is to determine how many apples are left in a basket after some have been eaten. The Chain of Thought reasoning would involve steps like calculating the original number of apples, subtracting the number eaten, and arriving at the final count.\n"
            "Statement: What are the advantages of using this approach?\n"
            "Reformulated Statement: What are the advantages of using Chain of Thought reasoning?<|endoftext|>\n"
            "\n"
            "Example 2\n"
            "Chat History:\n"
            "Human: What is the principle of stare decisis in law?\n"
            "AI: The principle of stare decisis refers to the legal doctrine of adhering to precedent, where courts are bound to follow the decisions of higher courts in similar cases to ensure consistency and predictability in the legal system.\n"
            "Human: Can you explain why it is important in the judicial process?\n"
            "AI: Stare decisis is important because it promotes legal stability, predictability, and fairness. It ensures that similar cases are treated alike, reduces judicial arbitrariness, and helps individuals and businesses understand their legal obligations.\n"
            "Statement: What are some exceptions to this principle?\n"
            "Reformulated Statement: What are some exceptions to the principle of stare decisis?<|endoftext|>\n"
            "\n"
            "Example 3\n"
            "Chat History:\n"
            "Human: What is a contract?\n"
            "AI: A contract is a legally enforceable agreement between two or more parties that creates mutual obligations. It requires an offer, acceptance, consideration, and the intention to create legal relations to be valid.\n"
            "Human: What happens if one party breaches a contract?\n"
            "AI: If one party breaches a contract, the other party may seek remedies such as damages, specific performance, or rescission, depending on the terms of the contract and the severity of the breach.\n"
            "Statement: Can you explain the difference between damages and specific performance?\n"
            "Reformulated Statement: Can you explain the difference between damages and specific performance as remedies for a contract breach?<|endoftext|>\n"
            "\n"
            "Example 4\n"
            "Chat History:\n"
            "Human: What is photosynthesis?\n"
            "AI: Photosynthesis is the process by which green plants, algae, and some bacteria convert sunlight, carbon dioxide, and water into glucose and oxygen. It takes place in the chloroplasts of cells and is essential for the survival of most life on Earth.\n"
            "Human: What are the main stages involved?\n"
            "AI: The main stages are the light-dependent reactions, where sunlight is absorbed to produce ATP and NADPH, and the Calvin cycle, where carbon dioxide is fixed into glucose using ATP and NADPH.\n"
            "Statement: What factors can affect this process?\n"
            "Reformulated Statement: What factors can affect the process of photosynthesis?<|endoftext|>\n"
            "\n"
            "Example 5\n"
            "Chat History:\n"
            "Human: What is intellectual property law?\n"
            "AI: Intellectual property law protects creations of the mind, such as inventions, literary and artistic works, designs, symbols, names, and images. It provides creators exclusive rights to use and benefit from their creations for a certain period.\n"
            "Human: What are the different types of intellectual property?\n"
            "AI: The main types are patents (inventions), copyrights (creative works), trademarks (brand identifiers), and trade secrets (confidential business information).\n"
            "Statement: How do trademarks differ from copyrights?\n"
            "Reformulated Statement: How do trademarks differ from copyrights under intellectual property law?<|endoftext|>\n"
            "\n"
            "Example 6\n"
            "Chat History:\n"
            "Human: Can you explain what a supply chain is?\n"
            "AI: A supply chain is a network of individuals, organizations, resources, and technologies involved in creating and delivering a product or service, from raw materials to the end customer.\n"
            "Human: What are the key components of a supply chain?\n"
            "AI: Key components include sourcing raw materials, manufacturing, warehousing, transportation, inventory management, and distribution.\n"
            "Statement: What are the main challenges in managing it?\n"
            "Reformulated Statement: What are the main challenges in managing a supply chain?<|endoftext|>\n"
            "\n"
            "Example 7\n"
            "Chat History:\n"
            "Human: What happens if someone breaches a contract?\n"
            "AI: If someone breaches a contract, remedies are available depending on the circumstances. Common remedies include damages, specific performance, or rescission. To help further, what remedy do you want to pursue in this situation?\n"
            "Statement: Monetary compensation.\n"
            "Reformulated Statement: The remedy the user wants to pursue is monetary compensation.<|endoftext|>\n"
            "\n"
            "Example 8\n"
            "Chat History:\n"
            "Human: What should I do if my landlord is not making necessary repairs?\n"
            "AI: In cases where a landlord fails to make necessary repairs, you may have legal options such as withholding rent, repairing and deducting costs, or seeking legal action.\n"
            "Human: Are there specific conditions for withholding rent?\n"
            "AI: Yes, withholding rent is typically allowed only after proper notice to the landlord and in jurisdictions where such action is legally permitted. Could you clarify which option you would like to consider?\n"
            "Statement: Withholding rent.\n"
            "Reformulated Statement: The option the user wants to consider is withholding rent.<|endoftext|>\n"
            "\n"
            "Example 9\n"
            "Chat History:\n"
            "Human: How can I protect my business name from being copied?\n"
            "AI: To protect your business name, you can register it as a trademark, ensure it's unique, and enforce your rights if infringement occurs. Would you like to proceed with trademark registration?\n"
            "Statement: Yes.\n"
            "Reformulated Statement: The user wants to proceed with trademark registration.<|endoftext|>\n"


            # "You are a question creator. You are given chats of an ai conversation along with a query. Your task is to rewrite it in form of a complete question. You will not answer the query but rather just pass it as it is. In case the query cannot be understood directly but requires context from the conversation, you will pick out the necessary points and add them to the query. At any cost do not answer the query and only state the reformulated query."
            "Chat History: \n"
        )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "\nStatement: {input}\nReformulated Statement:"),
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
