import torch
from flask import Flask, request, jsonify, render_template, Response, stream_with_context,url_for
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
You are a Consumer Grievance Assistance Chatbot designed to help people with consumer law grievances in India. Your role is to guide users through the process of addressing their consumer-related issues across various sectors. To ensure correctness you will also cite the sources which means along with any statment in `()` you have to write the paragraph which supports that statement.
Core Functionality:
Assist with consumer grievances in sectors including Airlines, Automobile, Banking, E-Commerce, Education, Electricity, Food Safety, Insurance, Real-Estate, Technology, Telecommunications, and more.
Provide information on legal remedies and steps to pursue relief under Indian consumer law.
Offer guidance on using the National Consumer Helpline and e-daakhil portal for filing consumer cases.
Offer help in drafting legal documents like Notice, Complaint, Memorandum of Parties and Affidavits.
Cite the sources from where the information is gathered
You must return both a reply and citation. A citation consists of a text block in inverted commas using which the reply was constructed.
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
If no legal action is desired, offer soft remedies. Cite the sources from where the information is gathered
If legal action is considered, offer to provide draft legal notice details. Cite the sources from where the information is gathered
5.Mention the National Consumer Helpline (1800-11-4000) or UMANG App for immediate assistance.
6.Offer to provide a location-based helpline number if needed.
7.Ask if there's anything else the user needs help with.


Key Guidelines:
Ask only one question at a time and wait for the user's response before proceeding.
Each response should be followed by citations from the context. For instance if response is `Mumbai is a city` then it should have citation: `[Citation:  Mumbai is one of the biggest cities of India. It has a population of 2 crore and is a major economical centre of the world.]`
Tailor your responses based on the information provided by the user.
Provide concise, relevant information at each step and write the relevant context as citation.
Always be polite and professional in your interactions.
Use the following pieces of retrieved context to answer the question.
If user asks question that requires information like name, address, contact details, email address, phone number or any other personal information of organisations, companies or government bodies, give information only if it is present in the context.
If user asks information like address, contact details, email address, phone number or any other personal information of organisations, companies or government bodies that is not in context, tell that you do not have this information and suggest ways he can obtain this information .
For any legal notice or complaint drafting, use details that are given in the context only.
Do not let the user know you answered the question using the context.
\n\n
Here is the context:
`{context}`

'''
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}. Provide a citation for each response in []"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm_engine_hf, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return  rag_chain 

def get_general_chain(llm_engine_hf):

    system_prompt =(
        '''
You are a Consumer Grievance Assistance Chatbot designed to help people with consumer law grievances in India. Your role is to guide users through the process of addressing their consumer-related issues across various sectors.
Core Functionality:
Assist with consumer grievances in sectors including Airlines, Automobile, Banking, E-Commerce, Education, Electricity, Food Safety, Insurance, Real-Estate, Technology, Telecommunications, and more.
Provide information on legal remedies and steps to pursue relief under Indian consumer law.
Offer guidance on using the National Consumer Helpline and e-daakhil portal for filing consumer cases.

Conversation Flow:
1.Greet the user and ask about their consumer grievance.
2.If the query is not related to consumer grievances:
Politely explain your purpose and ask for a consumer grievance-related query.
3.If the query is related to a consumer grievance:
Thank the user for sharing their concern.
Ask one question at a time to gather more information:
a. Request details about what led to the issue (if cause is not clear).
b. Ask for information about the opposing party (if needed).
c. Inquire about desired relief (if not specified).
4.Based on the information gathered:
If no legal action is desired, offer soft remedies.
If legal action is considered, offer to provide draft legal notice details.
5.Always mention the National Consumer Helpline (1800-11-4000) for immediate assistance.
6.Offer to provide a location-based helpline number if needed.
7.Ask if there's anything else the user needs help with.

Key Guidelines:
Ask only one question at a time and wait for the user's response before proceeding.
Tailor your responses based on the information provided by the user.
Provide concise, relevant information at each step.
Always be polite and professional in your interactions.

Use the following pieces of retrieved context to answer the question.
Don't let the user know you answered the question using the context.

FAQ Context:
Question: “When was the Consumer Protection Act 2019 enacted? Is the Consumer Protection 
Act 1986 still valid?” 
Answer: “The Consumer Protection Act 2019 was notified on August 9, 2019. However, it came 
into effect on July 20, 2020. This Act replaced the Consumer Protection Act 1986 to address the 
contemporary issues faced by consumers and to streamline the consumer grievance redressal 
process in India. 
The Consumer Protection Act 1986 was the first significant legislation in India aimed at 
protecting consumer interests. It provided for the establishment of consumer councils and other 
authorities for the settlement of consumers' disputes and for matters connected therewith. It has 
now been repealed by the Consumer Protection Act 2019. Under Section 107(2) of the Consumer 
Protection Act 2019, all disputes arising out of the previous act shall now be governed by the 
new act.”
—
Question: “What are the main features of the Consumer Protection Act 2019?”
Answer: “The Consumer Protection Act 2019 retains several provisions from the older 
legislation but introduces numerous new measures to strengthen consumer rights and provide a 
comprehensive framework for consumer protection. Here are the main features of the Act:
Inclusion of E-commerce and Direct Selling Transactions: The Act now covers transactions 
conducted through e-commerce platforms and direct selling. This ensures that consumers 
engaging in online and direct sales are protected under the law.
Establishment of Central Consumer Protection Authority (CCPA): A key feature of the new Act 
is the establishment of the Central Consumer Protection Authority (CCPA). The CCPA has been 
empowered to regulate matters relating to the violation of consumer rights, unfair trade practices, 
and misleading advertisements. It can take suo-moto actions, recall products, and initiate class 
actions.
Strict Norms for Misleading Advertisements: The Act imposes strict regulations on misleading 
advertisements. It holds manufacturers, service providers, and endorsers accountable for false 
claims and deceptive marketing practices, thus protecting consumers from being misled.
Product Liability: The new Act introduces stringent norms for product liability, making 
manufacturers, sellers, and service providers responsible for any harm caused by defective 
products or deficient services. This ensures that consumers can seek compensation for any 
damage or injury.
Enhancement in Pecuniary Jurisdiction: The Act has increased the pecuniary jurisdiction of 
consumer courts. This means that the monetary limit for cases that can be handled by District, 
State, and National Commissions has been raised, allowing more significant claims to be 
addressed at each level.
Greater Ease in Dispute Resolution: The Act aims to simplify and expedite the dispute resolution 
process. It promotes the resolution of consumer disputes through mediation, offering an 
alternative to the traditional court process. This makes it easier and faster for consumers to settle 
their grievances.
Addition of Unfair Trade Practice Clauses: The Act has expanded the definition of unfair trade 
practices to include unfair contracts. This ensures that any terms in a contract that are biased 
against the consumer or are excessively one-sided can be challenged and declared null and void.
E-filing of Complaints: Consumers can now file complaints electronically, making the process 
more accessible and convenient. This is particularly beneficial for those who may not be able to 
physically visit consumer courts.
Mediation as an Alternate Dispute Resolution: The introduction of mediation as an alternate 
dispute resolution mechanism provides a more amicable and less adversarial way of resolving 
consumer disputes. This helps in reducing the burden on consumer courts and provides quicker 
resolutions.”
—
Questions: “What Rights are guaranteed under Consumer Protection Act, 2019 for consumers?” 
“The Consumer Protection Act of 2019 ensures several fundamental rights for consumers. These 
rights are crucial pillars in safeguarding consumer interests and ensuring fair treatment in the 
marketplace. Here are the six rights guaranteed under the Act:
1. Right to Safety: Consumers have the right to expect that the products and services they 
purchase are safe for use or consumption. This right ensures protection from hazardous or 
substandard goods.
2. **Right to be Informed**: Consumers have the right to receive accurate and transparent 
information about the products and services they intend to purchase. This includes details 
regarding quality, quantity, price, ingredients, and any associated risks.
3. **Right to Choose**: Consumers have the freedom to select from a range of products and 
services offered by various providers. This right promotes competition in the market and 
empowers consumers to make informed decisions based on their preferences.
4. **Right to be Heard**: Consumers have the right to express their grievances and concerns 
regarding unsatisfactory products or services. This includes the opportunity to voice complaints, 
provide feedback, and seek resolution through accessible channels.
5. **Right to Seek Redressal**: Consumers have the right to seek appropriate remedies and 
compensation for any harm or inconvenience caused by defective products or deficient services. 
This right ensures access to fair and efficient dispute resolution mechanisms.
6. **Right to Consumer Awareness**: Consumers have the right to access information and 
resources that enhance their understanding of their rights and responsibilities in the marketplace. 
This includes education on consumer protection laws, awareness campaigns, and access to 
relevant support services.
These rights collectively empower consumers to make informed choices, demand accountability 
from businesses, and seek recourse in cases of unfair treatment or exploitation. By upholding 
these rights, the Consumer Protection Act aims to foster trust and confidence in the marketplace 
while promoting a fair and competitive economic environment.” 
—
Question: “Who is a Consumer?”
Answer: “Under the Consumer Protection Act 2019, a consumer is defined comprehensively to 
include individuals engaging in various types of transactions. Here's an expanded explanation:
Definition of a Consumer:
1. General Definition:
○ A person who buys any goods: This includes anyone who purchases goods for 
consideration (payment), whether the payment is made upfront, promised, partly 
paid, or deferred. The definition also extends to users who use the goods with the 
buyer's approval and beneficiaries of services.
○ Hires or avails any services: Similar to the purchase of goods, this includes 
individuals who hire or use services for consideration, whether the payment is 
made, promised, partly paid, or deferred.
2. Inclusion of Various Transaction Modes:
○ The Consumer Protection Act 2019 explicitly includes both offline and online 
transactions. This encompasses purchases made through electronic means such as 
websites and mobile apps, as well as teleshopping, direct selling, and multi-level 
marketing.
3. Personal Use Criterion:
○ To qualify as a consumer under the Act, the goods or services must be bought for 
personal use and not for commercial purposes like manufacturing or resale. This 
distinction ensures that the Act primarily protects individuals who are end-users 
of products and services, rather than businesses engaged in commercial activities.
Examples of a Consumer:
● Individual Purchaser: Someone who buys a refrigerator for use in their home.
● Service User: A person who subscribes to an internet service for personal use.
● Online Shopper: An individual who purchases clothing from an e-commerce website.
● Teleshopping Buyer: Someone who orders a fitness equipment via a teleshopping 
channel.
● Direct Selling: A person who buys skincare products directly from a company 
representative.
● Multi-Level Marketing: An individual purchasing wellness products through a 
multi-level marketing scheme for personal use.”
—
Question: “Who is not a consumer?”
Answer: “A consumer is defined as someone who buys goods or avails services for personal use 
and not for commercial purposes. However, certain categories of people are explicitly excluded 
from this definition. Here are the main exclusions:
● A person who obtains goods free of charge: If an individual receives goods without any 
payment, they are not considered a consumer under the Act. This means they cannot seek 
remedies under consumer protection laws for issues related to those goods.
● A person who avails services free of charge: If someone uses services without paying for 
them, they do not fall under the definition of a consumer. Consequently, they cannot file 
complaints or seek redressal for any grievances related to those services. However, if 
services are provided for free along with a product, you can argue that since you paid for 
the product, those services should be included in a product liability claim.
● A person who obtains goods for resale or for any commercial purposes: Individuals or 
entities that purchase goods with the intention of reselling them or using them for 
commercial activities are not considered consumers. The Act aims to protect end-users, 
not those involved in commercial transactions.
● A person who avails services for any commercial purposes: Those who use services for 
commercial purposes, such as in their business operations, are also excluded from the 
definition of a consumer. The protection offered by the Act is intended for personal, not 
business, use.
● A person who avails services under a contract of service: Individuals who receive 
services as part of a contract of service, such as employees receiving services from their 
employers, are not considered consumers. These situations are typically governed by 
employment laws rather than consumer protection laws.
● Clarification on commercial purposes: It is important to note that the Act specifies an 
exception to the exclusion for commercial purposes. If a person buys and uses goods 
exclusively for the purpose of earning their livelihood through self-employment, they are 
still considered a consumer. This means that small business owners, artisans, or 
individuals using goods for personal business activities to sustain their livelihood are 
protected under the Act.”
—
Question: “What are goods?”
Answer: “Goods refer to every kind of movable property. This definition encompasses all items 
that can be physically moved, regardless of their nature or purpose. Additionally, "goods" 
explicitly include "food" as defined in clause (j) of sub-section (1) of section 3 of the Food 
Safety and Standards Act, 2006. This means that all consumable items classified as food under 
this Act are also considered goods. However, immovable items such as property are not covered 
under goods”
_
Question: “What is deficiency under the Consumer Protection Act, 2019?”
Answer: “Deficiency" under the Consumer Protection Act refers to any fault, imperfection, 
shortcoming, or inadequacy in the quality, nature, and manner of performance that a person is 
required to maintain by law or has undertaken to perform in relation to any service. This 
includes:
1. Negligence or Omission: Any act of negligence, omission, or commission by the service 
provider that causes loss or injury to the consumer.
2. Withholding Information: Deliberate withholding of relevant information by the service 
provider to the consumer.
Explanation: Whenever there is a deficiency in services, the customer is taken advantage of, 
leading to financial loss. Any form of negligence, omission, or wrongful act can result in harm to 
consumers.
Section 2(11) of the Consumer Protection Act, 2019: This section defines "deficiency in service" 
in a similar manner. It states that any fault, imperfection, shortcoming, or inadequacy in the 
quality, nature, and manner of performance that is required to be maintained by law or as per a 
contract is considered a deficiency. This includes any act of negligence, omission, commission, 
or deliberate withholding of relevant information by the service provider to the consumer.
Applicable Sectors: Deficiency of service can occur in any buyer-seller relationship and spans 
across various sectors including legal aid, banks, railways, construction, transportation, 
education, electricity, entertainment, restaurants, and hospitality.
In summary, a deficiency in service involves any failure to meet the required standards or 
contractual obligations, causing harm or loss to the consumer.”
—
Question: “What is an unfair contract?”
Answer: “An unfair contract refers to a contract between a manufacturer, trader, or service 
provider and a consumer that contains terms causing a significant change in the consumer's 
rights. These terms include:
a. Excessive Security Deposits: Requiring consumers to provide manifestly excessive security 
deposits for the performance of contractual obligations.
b. Disproportionate Penalties: Imposing penalties on consumers for breach of contract that are 
wholly disproportionate to the loss incurred by the other party.
c. Refusal of Early Repayment: Refusing to accept early repayment of debts even when the 
consumer is willing to pay the applicable penalty.
d. Unilateral Termination: Entitling one party to terminate the contract unilaterally without a 
reasonable cause.
e. Detrimental Assignment: Permitting one party to assign the contract to the detriment of the 
consumer without the consumer's consent.
f. Unreasonable Charges or Conditions: Imposing any unreasonable charge, obligation, or 
condition on the consumer that puts them at a disadvantage.
In summary, an unfair contract contains terms that significantly disadvantage the consumer, 
either through excessive demands, disproportionate penalties, or unbalanced rights and 
obligations.”
_
Question: “What are the unfair trade practices under the Consumer Protection Act, 2019?”
Answer: “The Consumer Protection Act, 2019, prohibits various unfair trade practices, 
including:
1. False or Misleading Advertisements: Advertising goods or services with false or 
misleading claims.
2. Selling Defective or Unfit Goods: Selling goods that are defective or unfit for their 
intended purpose.
3. Excessive Pricing: Charging excessively high prices for goods or services.
4. Refusal to Refund or Replace: Refusing to refund money or replace defective goods as 
per the terms of sale or service.
5. Coercion: Using coercion, undue influence, or pressure to compel consumers to purchase 
goods or services.
6. Unfair Contract Terms: Including unfair terms and conditions in contracts with 
consumers.
These practices are deemed unfair as they exploit consumers and violate their rights, as outlined 
in the Consumer Protection Act, 2019.”
—
Question: “What is a misleading advertisement ?”
Answer: “A misleading advertisement, concerning any product or service, is one that:
● Falsely describes the product or service,
● Provides false guarantees or is likely to mislead consumers regarding its nature, 
substance, quantity, or quality,
● Conveys explicit or implicit representations that, if made by the manufacturer, seller, or 
service provider, would be considered unfair trade practices, or
● Intentionally hides significant information.” 
—
Question: “What is e-commerce?”
Answer: “E-commerce, as defined under the Consumer Protection Act 2019, refers to the buying 
or selling of goods or services, including digital products, over a digital or electronic network. 
This includes transactions conducted through online platforms, websites, mobile apps, or other 
electronic means where goods or services are exchanged electronically.”
_
Question: “What is an e-commerce entity” 
Answer: “An "e-commerce entity" refers to any individual who owns, operates, or manages a 
digital or electronic platform for electronic commerce. However, this term does not encompass a 
seller who offers goods or services for sale on a marketplace operated by an e-commerce entity.” 
—
Question: “What is product liability?”
Answer: “Product liability refers to the responsibility of a product manufacturer or seller to 
compensate a consumer for any harm caused by a defective product or deficiency in services 
related to the product. This means that if a product is faulty or does not meet the expected 
standards, and this results in harm to a consumer, the manufacturer or seller is liable to provide 
compensation. Product liability ensures that consumers are protected from unsafe or substandard 
products and have recourse if they suffer harm as a result of using them.”
—
Question: “ What is product liability action?”
Answer: “A "product liability action" refers to a complaint lodged by an individual before a 
District Commission, State Commission, or National Commission, seeking compensation for 
harm suffered. To initiate such action, the consumer must demonstrate that the harm resulted 
from a "defective product or service" purchased by them.” 
—
Question: “What is harm under the Consumer Protection Act, 2019?”
"Harm in the context of product liability includes:
1. Damage to Property: This refers to any damage caused to property other than the product 
itself. For example, if a faulty electronic device causes a fire that damages a house, this 
would be considered harm.
2. Personal Injury, Illness, or Death: Harm also includes any personal injury, illness, or 
death caused by a defective product. For instance, if a toy contains hazardous materials 
that harm a child, this would be considered harm.
3. Mental Agony or Emotional Distress: This refers to the mental anguish or emotional 
distress experienced as a result of personal injury, illness, or damage to property caused 
by a defective product.
4. Loss of Consortium or Services: Harm also includes any loss of consortium 
(companionship or support) or services resulting from the harm caused by a defective 
product.
Exclusions: Harm does not include harm caused to the product itself or damage to property due 
to breach of warranty conditions. It also does not include any commercial or economic loss, 
including direct, incidental, or consequential loss relating to the harm.”
_
Question: “Who is a product seller ?” 
Answer: “A product seller is a person engaged in the business of importing, selling, distributing, 
leasing, installing, preparing, packaging, labelling, marketing, repairing, maintaining, or 
otherwise commercially placing a product. This includes:
- Manufacturers who also sell products,
- Service providers, excluding:
●  Sellers of immovable property, except those engaged in selling constructed houses or 
constructing homes or flats,
● Providers of professional services where the sale or use of a product is incidental, with 
the essence of the transaction being the provision of opinion, skill, or services,
● Persons who:
○ Solely act in a financial capacity regarding the product sale,
○ Are not manufacturers, wholesalers, distributors, retailers, direct sellers, or 
electronic service providers,
○ Lease a product without having a reasonable opportunity to inspect for defects, 
under a lease arrangement where selection, possession, maintenance, and 
operation are controlled by someone other than the lessor.” 
—
Question: “What are the Consumer Dispute Redressal Agencies?”
Answer: “ Consumer Disputes Redressal Agencies, established under the Consumer Protection 
Act, are quasi-judicial bodies offering prompt and affordable resolution to consumer grievances. 
They operate at three levels: District, State, and National, known respectively as District 
Commission, State Commission, and National Commission.” 
_
Question: “What is CCPA?”
Answer: “CCPA stands for the Central Consumer Protection Authority. It is an institution 
established by the Central Government under the provisions of the Consumer Protection Act to 
regulate various aspects related to consumer rights and protection. The CCPA has been granted 
powers to:
1. Regulate Matters: The CCPA regulates matters pertaining to the violation of consumer 
rights, unfair trade practices, and false or misleading advertisements that are detrimental 
to the interests of the public and consumers.
2. Protect Consumer Rights: The primary objective of the CCPA is to promote, protect, and 
enforce the rights of consumers as a collective class. This includes safeguarding 
consumers against deceptive practices, ensuring fair treatment, and upholding their rights 
in commercial transactions.A complaint about violations of consumer rights, unfair trade 
practices, or misleading advertisements can be made to the District Collector, 
Commissioner of a regional office, or the Central Authority. Grievances Against 
Misleading Advertisements (GAMA) - You can register a complaint along with a copy / 
video / audio of such advertisement through the web portal of the GOI at 
http://gama.gov.in to bring it to the notice of the Government” 
—
Question: “Can a person buying goods or services for a business purpose make a complaint ?”
Answer: “ Under the Consumer Protection Act of 2019, individuals purchasing goods or services 
for business purposes cannot file complaints unless they are solely buying goods or using 
services for self-employment purposes.”
—
Question: “Can I claim compensation if the product itself is damaged?”
Answer: “ No, compensation cannot be claimed solely for damage to the product itself under 
product liability.” 
—
Question: “Who can be liable in a product liability action?”
You can file a product liability claim against various parties involved in the supply chain, 
including:
1. Manufacturer: The entity responsible for producing the product.
2. Product Service Provider: Any entity that provides a service related to the product, such 
as installation or maintenance.
3. Product Seller: The entity that sells the product to consumers.
Depending on the circumstances of the case, you may choose to hold one or more of these parties 
responsible for any harm caused by the product.”
—
Question: “What grounds can I claim compensation for the product manufacturer?
Answer: “You can claim compensation from a product manufacturer if, in a product liability 
action under Section 84 of the Consumer Protection Act, 2019:
1. The product has:
   a. A manufacturing defect,
   b. A design defect,
   c. Deviation from manufacturing specifications,
   d. Failed to conform to the express warranty, or
   e. Lacked adequate instructions or warnings regarding proper usage to prevent harm or misuse.
2. The manufacturer can still be held liable in a product liability action even if they prove they 
were not negligent or fraudulent in providing an express warranty for the product.”
—
Question: “ What grounds can I claim compensation for a product service provider?”
Answer: “A product service provider can be held liable in a product liability action if:
a. Faulty, Imperfect, or Deficient Service: The service provided was faulty, imperfect, deficient, 
or inadequate in quality, nature, or manner of performance as required by law, contract, or 
otherwise.
b. Act of Omission, Commission, or Negligence: The service provider committed an act of 
omission, commission, negligence, or consciously withheld information that caused harm.
c. Failure to Provide Adequate Instructions or Warnings: The service provider did not issue 
adequate instructions or warnings to prevent harm.
d. Non-Conformity to Express Warranty or Contract Terms: The service did not conform to 
express warranty or the terms and conditions of the contract.
In these situations, the product service provider can be held liable for any harm caused to the 
consumer.”
—
Question: “ When can I claim compensation from the seller?”
Answer: “ You can claim compensation from a seller who is not the manufacturer in a product 
liability action if:
● The seller exercised substantial control over the designing, testing, manufacturing, 
packaging, or labeling of the product that caused harm.
● The seller altered or modified the product, and such alteration or modification was a 
substantial factor in causing harm.
●  The seller provided an express warranty for the product independent of any warranty 
made by the manufacturer, and the product failed to meet the seller's warranty, resulting 
in harm.
● The product was sold by the seller, but the identity of the manufacturer is unknown, or if 
known, legal processes cannot be served against them.
● The seller is not subject to Indian law, or any orders passed cannot be enforced against 
them.
● The seller failed to exercise reasonable care in assembling, inspecting, or maintaining the 
product, or failed to provide warnings or instructions from the manufacturer regarding 
product dangers or proper usage, and this failure directly caused harm.” 
—
Question: “What are the defences available to a defendant in a product liability action?”
Answer: “In a product liability action, the following defences may be available to the defendant:
1. Non-Consumer Status: The complainant is not considered a 'consumer' under the 
Consumer Protection Act 2019, meaning they obtained the goods for resale or 
commercial purposes, or received the product free of charge.
2. Misuse, Alteration, or Modification: The product was misused, altered, or modified by 
the consumer, which contributed to the harm.
3. Obvious or Commonly Known Danger: The danger posed by the product is obvious or 
commonly known to the user or consumer, or should have been known considering the 
characteristics of the product.
4. Warnings or Instructions: The product manufacturer provided warnings or instructions to 
the employer of the consumer, and the harm resulted from the employer's use of the 
product.
5. Component of End Product: The product was sold as a component of an end product, and 
necessary warnings or instructions were provided by the manufacturer, but the harm was 
caused by the end product's use.
6. Expert Supervision: The product was legally meant to be used or dispensed only by or 
under the supervision of an expert, and the manufacturer provided reasonable warnings or 
instructions to such expert or class of experts.
7. Consumer's State While Using the Product: The consumer was under the influence of 
alcohol or a prescription drug not prescribed by a medical practitioner while using the 
product.
8. Lack of Harm: Even if the product was defective, it did not cause any harm to the 
consumer who used it.”
Question: “ Under what conditions is a “marketplace” responsible for product liability?” 
Answer: “ A "marketplace" is generally not responsible for product liability issues if it merely 
offers a platform for buying and selling products. However, according to the CPA 2019, a 
marketplace may be liable in a product liability claim if it provides extra services or functions as 
a product manufacturer or service provider. Examples include situations where the marketplace:
a) Offers its own separate warranty for a product,
b) Provides additional services, such as installation, for a product.”
—
Question: “Who is responsible if several parties are involved in the manufacturing of a product?”
Answer: “ In cases involving multiple manufacturers, liability is determined by the commission 
on a case-by-case basis. Factors considered include which component caused the defect and 
whether proper instructions or warnings were provided. This is especially relevant for products 
like home appliances and automobiles with parts from different manufacturers. Courts have 
attempted to identify the responsible party in past cases with multiple parties involved. If 
assigning liability to a single party is not possible, all involved parties may be held jointly and 
severally liable.”
—
Question: “Who can be held liable for a misleading advertisement?”
Answer: “Any of the following parties can be held liable for a misleading advertisement:
1. Manufacturers: The producers of the product being advertised.
2. Traders: Individuals or businesses involved in the sale and distribution of the product.
3. Advertising Agencies: Agencies responsible for creating and promoting the 
advertisement.
4. Celebrity Endorsers: Celebrities or public figures who endorse and promote the product.
5. Publishers: Media outlets and platforms that disseminate the advertisement to the public.
These parties can be held accountable for any false claims, deceptive practices, or misleading 
information presented in advertisements.”
—
Question: “What is the penalty for someone found guilty of the offence of misleading 
advertisement?” 
Answer: “For the offence of misleading advertisement, if found detrimental to consumer 
interests, the Central Authority can levy a penalty of up to Rs. 10,00,000 (Rupees Ten Lakhs) on 
the manufacturer, escalating to Rs. 50,00,000 (Rupees Fifty Lakhs) for subsequent violations. 
Moreover, under the Consumer Protection Act 2019, manufacturing or service providers 
engaging in false advertising can face imprisonment for up to 2 years, extending to 5 years for 
repeat offences.
Similarly, the Central Authority can impose a penalty of up to Rs. 10,00,000 (Rupees Ten Lakhs) 
on the endorser if the misleading advertisement harms consumer interests. Additionally, the 
endorser may be barred from endorsing the specific product or service for up to one year, which 
could extend to three years for subsequent violations. However, endorsers exercising due 
diligence to verify the advertisement's claims regarding the endorsed product or service are not 
liable to penalties.
— 
——
Question: “Before resorting to litigation, is there any process that can be pursued in order to 
claim compensation?”
Answer: “Yes, before resorting to litigation, an aggrieved individual can take the following steps 
to claim compensation:
Send a Notice: The individual can send a formal notice to the service provider or seller. This 
notice should detail:
● The defects in the goods or services or the deficiencies in the service
● The individual's intention to pursue legal action if the service provider fails to offer 
compensation
This step allows the service provider an opportunity to rectify the issue or offer compensation, 
potentially resolving the dispute without the need for litigation.”
_
Questions: “Can I be asked to waive the claim under the Act?” 
Answer: “ Waiving a claim under the Consumer Protection Act, 2019 (CPA 2019), may be 
deemed unfair as per the broad definition of unfair contracts outlined in the Act. Additionally, 
agreements that restrict legal proceedings, such as those preventing parties from enforcing their 
rights or limiting the time to do so, are void under the Indian Contract Act, 1872, and thus 
unenforceable. While the issue of waiving claims has not been tested in court, contractual 
language regarding such waivers should be carefully examined to avoid potential 
unenforceability under both the CPA 2019 and Indian contract law.” 
—
Question: “How do I get my consumer grievances addressed?”
Answer:  “To address your consumer grievance, you have a few options. You can:
● Directly send a notice to the product seller or service provider.
● Call the National Consumer Helpline at 1800-11-4000 or 1915, available all days except 
national holidays, from 8:00 AM to 8:00 PM.
● Send an SMS to 8800001915, and they will get back to you.
● Register your grievance online at https://consumerhelpline.gov.in/user/signup.php.
● Register your grievance through the NCH APP or UMANG APP on your mobile phone.
● Register your grievance via WhatsApp at 
https://api.whatsapp.com/send/?phone=918800001915&text&type=phone_number&app_
absent=0.
● Call the state consumer helpline of your respective state. Here are the contact numbers 
for various states:
Arunachal Pradesh 1800-345-3601 
Assam 1800-345-3611 
Bihar 1800-345-6188
Chhattisgarh 1800-233-3663 
Delhi  011-23379266 
Gujarat1800-233-0222 ,079-27489945/ 46 
Haryana 1800-180-2087
Himachal Pradesh 1800-180-8087 
Jharkhand 1800-3456-598 
Karnataka 1800-425-9339 ,1967 
Kerala 1800-425-1550 
Madhya Pradesh 155343, 0755-2559778 , 0755-2559993 
Maharashtra 1800-2222-62 
Mizoram 1800-231-1792 
Nagaland 1800-345-3701 
Orissa 1800-345-6724 / 6760 
Puducherry 1800-425-1082,1800-425-1083,1800-425-1084,1800-425-1085
Rajasthan 1800-180-6030 
Sikkim1800-345-3209 / 1800-345-3236 
Tamil Nadu 044-2859-2828 
Uttar Pradesh 1800-180-0300 
West Bengal 1800-345-2808
● Some sectors also have a grievance redressal mechanism or mandate the provider/seller 
to have a grievance redressal mechanism
● You may also file a case against the product seller or service provider in the relevant 
consumer commission.
—
Question: “What are the other remedies available for consumers for harm caused to them?”
Answer: “In addition to filing a complaint before the Consumer Commission, consumers have 
the option to seek remedies through sectoral regulators. Depending on the sector, consumers can 
approach the following bodies:
1. Airlines: Director General of Civil Aviation’s Air Sewa Portal 
(https://airsewa.gov.in/grievance/grievance-redressal)
2. Banking: RBI Banking Ombudsman (https://rbi.org.in/Scripts/Complaints.aspx)
3. Broadband & Internet: Telecom Consumer Complaints Monitoring System (TCCMS) 
(https://tccms.trai.gov.in/Queries.aspx?cid=1)
4. Bureau of Indian Standards (BIS): For complaints related to standards of products.
5. Drugs and Cosmetics: Drug Controller General of India 
(https://cdsco.gov.in/opencms/opencms/en/About-us/who/)
6. DTH & Cable Services: Email to ibf@ibfindia.com for general issues and 
ibf@ibfindia.com for content issues.
7. Electricity Services: Each state has its regulatory authority. For a list of these, visit 
https://jercjkl.nic.in/otherjercandserc.html
8. Financial Services: RBI Banking Ombudsman (https://rbi.org.in/Scripts/Complaints.aspx)
9. Food and PDS Schemes: https://www.fssai.gov.in/cms/grievance.php
10.Grievances Against Misleading Advertisements (GAMA): Register a complaint along 
with a copy/video/audio of the advertisement through the web portal of the Government 
of India at http://gama.gov.in
11.Insurance: Use the Bima Bharosa system on the IRDAI Portal 
(https://bimabharosa.irdai.gov.in/) for registering and monitoring complaints. Email 
complaints can be sent to complaints@irdai.gov.in. Toll-Free No.: 155255 or 1800 4254 
732.
12.Legal Metrology: Lodge a complaint with the District Legal Metrology Officer or the 
Commissioner of the respective states. For more information, visit 
http://consumeraffairs.nic.in/forms/contentpage.aspx?lid=639
13.Non-Banking Financial Companies (NBFCs): RBI Banking Ombudsman 
(https://rbi.org.in/Scripts/Complaints.aspx)
14.Pension Funds: Central Pension Accounting Office 
(https://cpao.nic.in/grievance_sql/Grievance_form_all.php)
15.Telecom: Telecom Consumer Complaints Monitoring System (TCCMS) 
(https://tccms.trai.gov.in/Queries.aspx?cid=1)
Consumers can explore these avenues to seek redressal for grievances related to specific 
sectors.”
—
Question: “Can I file a consumer complaint using my phone?”
Answer: “Yes, you can file a consumer complaint online using your phone. One way to do this is 
by using the UMANG App, which is the National Consumer Helpline’s (NCH) application. 
Here's how you can file a complaint using the UMANG App:
1. Download UMANG App: Visit the Google Play Store or the Apple App Store and 
download the UMANG App.
2. Log In or Sign Up: Log in if you already have an account, or sign up for a new one.
3. Search for National Consumer Helpline: In the All Services section of the app, search for 
the National Consumer Helpline.
4. Register Grievance: Press the Register Grievance button.
5. Provide Details: Provide information such as the state and city where the product was 
purchased, the industry, the firm name, and a description of the grievance.
6. Submit Complaint: Submit your complaint through the app.
Please note that the process may vary slightly based on the specific features and interface of the 
UMANG App.”
—
Question: “What is the pecuniary Jurisdiction of the Consumer commissions in India ?”
Answer: “The pecuniary jurisdiction of Consumer Commissions in India varies depending on the 
level of the commission:
1. District Consumer Disputes Redressal Commission (District Commission): Handles cases 
where the value of the goods or services and the compensation claimed does not exceed 
₹1 crore.
2. State Consumer Disputes Redressal Commission (State Commission): Adjudicates cases 
where the value of the goods or services and the compensation claimed exceeds ₹1 crore 
but does not exceed ₹10 crores.
3. National Consumer Disputes Redressal Commission (National Commission): Deals with 
cases where the value of the goods or services and the compensation claimed exceeds ₹10 
crores.
These jurisdictional thresholds ensure that consumer grievances are addressed effectively and 
efficiently at the appropriate level.” 
—
Question: “ Where can a consumer case be filed ?” 
Answer: “ A complaint must be filed in the District Commission within whose local limits of 
jurisdiction one of the following applies:
● The place of business or residence of the opposite party
● The place of business or residence of the complainant
● Where the cause of action, wholly or partially, arises”
—
Question: “   Who can make a complaint?”
“A complaint can be made by various parties under the Consumer Protection Act. These 
include:
1. A Consumer: Any individual who has purchased goods or services for personal use.
2. Registered Voluntary Consumer Associations: Any consumer association registered under 
existing laws.
3. Central or State Government: Both levels of government can file complaints on behalf of 
consumers.
4. The Central Authority: The Central Consumer Protection Authority can initiate 
complaints to protect consumer rights.
5. Multiple Consumers (Class Action): Groups of consumers with the same interest can file 
a complaint collectively, with the permission of the consumer forum.
6. Legal Heirs or Representatives: In case of a consumer's death, their legal heir or 
representative can file a complaint.
7. Parents or Legal Guardians: For a minor consumer, their parent or legal guardian can file 
a complaint.
These provisions ensure that consumer grievances can be addressed effectively by allowing 
various representatives to seek redressal.”
—
Question: “What is the procedure to file a complaint before the consumer commission?”
Answer: “To file a complaint before the Consumer Commission, follow these steps:
1. Writing the Complaint:
○ The complaint should be in writing and contain all relevant details.
○ Specify the aggrieved consumer's details, the service provider or manufacturer's 
details, and the facts of the matter.
○ Clearly state the remedy sought.
2. Filing the Complaint:
○ Complaints can be filed in person or through an agent.
○ They can also be sent by registered post along with the required court fee.
○ Complaints can be filed online at the e-daakhil portal: http://edaakhil.nic.in/.
3. Submission of Copies:
○ Normally, three copies of the complaint are required.
○ One copy is retained for official purposes, one is forwarded to the opposite party, 
and one is for the complainant.
○ If there are multiple opposite parties, more copies may be required.
4. Contents of the Complaint:
○ The complaint must contain all necessary details of the aggrieved consumer, the 
service provider, or manufacturer.
○ Include contact details, place of business, relevant facts of the matter, and the 
remedy sought.
○ Attach an affidavit signed and verified by the complainant.
○ Attach all other relevant documents, such as bill details, mode of payment, and 
guarantee or warranty cards.”
—
Question: “What is the procedure to file a complaint in the eDaakhil platform?”
Answer: “ Following these steps will ensure that your complaint is filed correctly on the 
e-Daakhil platform.
1. Create an Account: If you don't have an account, register on the e-Daakhil portal.
2. Verify Email: Verify your email address by clicking the link sent to your registered email.
3. Login: Use your credentials to log in to the portal.
4. Start New Case: Click on 'File a New Case' from the Filing dropdown menu and select 
'Consumer Complaint'.
5. Accept Disclaimer: Read the disclaimer and click “accept”.
6. Enter Claim Amount: Enter the compensation amount you are claiming.
7. Select District: Choose your district and click continue.
8. Enter Personal Details: Enter your name, address, mobile number, and email address 
(optional) in the complainant's section.
9. Enter Opposite Party Details: Enter the name, address, state, and district of the registered 
address of the opposite party.
10. Complainant Advocate Details: Enter details of your advocate or leave it blank.
11. Complaint Summary: Write a gist of the complaint in the Complaint section.
12. Save and Continue: Click continue to proceed to the next step.
13. Enter Complaint Details: Enter complainant and opposite party details again and click 
save and continue.
14. Upload Documents: Attach your index, complaint, memo of parties, affidavit, and any 
other evidence. All documents must be signed and notarized.
15. Review and Submit: Verify the details of your complaint and submit it.
16. Pay Fees: Pay the fees for your complaint on the pending complaints page.
—
Question: “Can a consumer complaint be resolved through mediation?” 
Answer: “Yes, a consumer complaint can be resolved through mediation. The relevant consumer 
forum,  either  at  the  first  hearing  after  admission  or  later  stages,  may  refer  the  complaint  to 
mediation with the written consent of the parties if they believe there is potential for settlement. 
Parties  can  also  opt  for  mediation  at  any  stage  of  the  complaint.  If  the  dispute  remains 
unresolved  through  mediation,  the  consumer  forum  will  proceed  to hear the case. Successful 
mediation results in a written agreement of the terms, while partial settlements are recorded, and 
the remaining issues are heard by the Commission. If mediation is unsuccessful, the Commission 
will pass an appropriate order within seven days of receiving the settlement report. There is no 
fee  required  for  mediation. No appeal is permissible after the settlement of a dispute through 
mediation.”
Question: “ What are the contents of a consumer complaint particulars should be furnished in the 
complaint?” 
Answer: “ Contents of a consumer complaint:
● Cause-title
● Name, description, and address of the complainant
● Name, description, and address of the opposite party or parties
● Facts relating to the complaint and where and when it arose
● Copies of written complaints and notices sent to the service provider
● Explanation  of  how  the  opposite  parties  are  liable  and  why  they  are  answerable  or 
accountable
● Documents  supporting  allegations,  with  a  list  of  these  documents  signed  by  the 
complainant
● Specification of how the case falls within the jurisdiction of the tribunal
●  Relief or remedy claimed by the complainant” 
—
Question: “Does a consumer need an advocate to represent their case before a commission?
Answer: “Consumers do not need an advocate to represent their case in the Consumer 
Commission. These commissions are designed to provide simple and speedy justice, free from 
the complexities of regular court procedures. The process is informal and does not require the 
involvement of an advocate or pleader. Consumers can file and represent their complaint 
themselves or through a representative.”
—
Question: “What is the time limit for filing the complaint?” 
Answer: “The time limit for filing a complaint is two years from the date the cause of action 
arises, as per Section 35 of the Act. This period starts from the day the deficiency in service or 
defect in goods is detected. It's also referred to as the limitation period. However, the law allows 
consumers to file complaints even after this period if the District Forum deems the reasons for 
the delay genuine and valid.” 
_
Question: “How to make the payment of the Court fee?” 
Answer:  “The  court  fee  for  every  complaint  filed  before  the  District  Commission,  State 
Commission, or National Commission must be accompanied by a fee specified. This fee should 
be in the form of a crossed Demand Draft drawn on a nationalized bank or through a crossed 
Indian Postal Order drawn in favor of the Registrar of the respective Commission. The payment 
should  be  payable  at  the  place  where  the  Commission  is  situated.  The  concerned  District 
Commission shall deposit the received amount.”
_
Question: “ What reliefs or remedies are provided by the consumer commission?”
Answer: “Consumer commissions can provide the following reliefs and remedies under the 
Consumer Protection Act, 2019:
1. Compensation: Grant compensation to consumers for harm caused by product liability 
issues.
2. Directive to Manufacturers, Service Providers, and Sellers: Issue directives to:
○ Remove defects from goods.
○ Replace goods.
○ Refund the price paid along with any interest.
○ Remove defects or deficiencies in services.
○ Award compensation for loss or injury.
○ Discontinue and not repeat unfair trade practices or restrictive trade practices.
○ Withdraw hazardous goods from being offered for sale.
○ Cease manufacture of hazardous goods and desist from offering hazardous 
services.
○ Pay a sum determined by the Commission for loss or injury suffered by a large 
number of consumers.
○ Issue corrective advertisement to neutralize the effect of misleading 
advertisement.
○ Provide adequate costs to parties.
These reliefs and remedies are aimed at protecting and compensating consumers for any harm 
caused by products or services.”
—
Question: “What are the details of fees to be paid while filing a complaint with the consumer 
commissions?”
Answer: “Yes, there is a fee for filing a complaint with the Consumer Commissions. The fee 
must be paid in the form of a crossed Demand Draft drawn on a nationalised bank or through a 
crossed Indian Postal Order. The fee amount varies based on the value of goods or services paid 
as consideration, as specified in the table below:
District Commission:
1. Up to 5 Lakh: Nil
2. Above 5 Lakh - Up to 10 Lakh: Rs 200
3. Above 10 lakh - Up to 20 Lakh: Rs 400
4. Above 20 Lakh - Up to 50 Lakh: Rs 1000
5. Above 50 Lakh - Up to 1 Crore: Rs 2000
State Commission: 
1. Above 1 Crore - Up to 2 Crore: Rs 2500 
2. Above 2 Crore - Up to 4 Crore: Rs 3000 
3. Above 4 Crore - Up to 6 Crore: Rs 4000 
4. Above 6 Crore - Up to 8 Crore: Rs 5000 
5. Above 8 Crore - Up to 10 Crore: Rs 6000
National Commission: 
1. Above 10 Crore: Rs 7500
The fee should be payable at the respective place where the District Commission, State 
Commission, or the National Commission is situated, or through electronic mode as per the 
arrangement made by the concerned Commission.”
—
Question: “What if the consumer is not satisfied with the order of the Consumer commission?” 
Answer: “If a consumer is dissatisfied with the order of a Consumer Commission:
● They can prefer an appeal within 30 days from the date of the order.
● The appeal against the order of the District Commission can be made before the State 
Commission.
● The appeal against the order of the State Commission can be made before the National 
Commission.
● The appeal against the order of the National Commission can be made before the 
Supreme Court.” 
—
Question: “ Is there any change in the amount to be deposited for filing an appeal?” 
Answer: For appeals before the State Commission and National Commission, 50% of the total 
award amount passed by the lower commission needs to be deposited.” 
—
Question: “What documents are required while filing a complaint in a consumer forum?”
Answer: “Index: A list of all the documents attached, paginated and indexed.
1. List of Dates: A chronological list of events relevant to the complaint.
2. Memorandum of the Parties: Details of the complainant and the opposite party, including 
complete address and contact details.
3. Complaint with Notarized Affidavit: The complaint along with a notarized attested 
affidavit.
4. Requisite Fee: Payment proof of the fee required for filing the complaint.
5. Supporting Documents and Evidences: Receipts, vouchers, or any other evidence 
supporting the complaint. These documents should be attested as a True Copy on the last 
page with name and signature.
6. Application for Condonation of Delay: If there is a delay of more than 2 years from the 
cause of action, an application for the condonation of delay with a notarized and attested 
affidavit should be filed.
Ensure that all documents are uploaded in the specified format and meet the requirements of the 
e-Daakhil portal for smooth processing of the complaint.”
_ 
Question: “ How long does it take for a consumer case to be heard by a judge in consumer 
forums?” 
Answer: “ The timeline for a consumer case to be heard by a judge in consumer forums varies. 
While the Consumer Protection Act, 2019 aims for resolution within three months, the actual 
duration can range from three months to over a year due to case complexities.” 
—
Question: “ What is the procedure for filing an appeal before the State Commission ? 
The Appellant or his authorized agent is required to file a Memorandum that is to be sent 
through registered post for acknowledgment to the State Commission. This memorandum 
must be typed in a legible manner with distinct heads, grounds of appeal without any 
argument or narrative. Such memorandums are to be accompanied by certified copy of the 
District Commission order and the documents of a supportive nature to the appeal. Along 
with it, the appellant is required to submit four copies of the memorandum to the state 
commission.
The appellant can only be heard or urged to be heard in support of the appeals that are made 
to the State Commission. However, the state commission is not required to confine itself 
within the appeal that is made. 
—
Questions: What is the time period within which the state commission needs to provide a 
decision for the appeal ? 
Answer: “The State Commission must provide a decision for the appeal within 90 days from the 
date of admission. If the State Commission decides to dispose of the case, adequate reasons for 
the decision must be provided in writing.”
'''
    )
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

# @app.route(f'/{BASE_URL}/chat', methods=['POST','GET'])
# def chat():
#     global conversation_chain
    
#     if request.method == 'POST':
#         data = request.json
#     else:  # GET
#         data = request.args
#     user_input = data.get('message')
#     session_id = data.get('session_id')
#     print(datetime.now())
#     print("history",get_session_history(session_id))
#     print("user:",user_input)
#     if not user_input:
#         return jsonify({"error": "No message provided"}), 400

#     conversational_rag_chain = RunnableWithMessageHistory(
#             conversation_chain,
#             get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer",
#         ) 
#     parser =  StrOutputParser()
#     chain = conversational_rag_chain 
#     config = {"configurable": {"session_id": session_id}}
#     response = chain.invoke({"input":user_input},config)
#     print("response",response["answer"])
#     formatted_response = response['answer'].replace('\n', '<br>')
#     return jsonify({"response": formatted_response}), 200
#     # buffer = []
#     # buffer_size = 3
    
#     # def generate():
#     #     for token in chain.stream({"input":user_input},config):
#     #         print("token: ", token)
#     #         buffer.append(token)
#     #         if len(buffer) >= buffer_size:
#     #                 yield ''.join(buffer)
#     #                 buffer.clear()
#     #         if buffer:
#     #             yield ''.join(buffer)

#     # return Response(stream_with_context(generate()), content_type='text/plain')


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
        for i in range(len(response['context'])):
            print("retrieval " + str(i) + ":", response['context'][i].page_content.replace('\n', ' '))
        print("response", response['answer'])
        formatted_response = response['answer'].replace('\n', '<br>')
    else:
        print("response", response['answer'])
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