import pandas as pd
import os
from PyPDF2 import PdfReader
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.transform import TransformChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from datetime import datetime
import copy
import time 

start_time = datetime.now()
print("Start Time: ", start_time)

def disp_dict(arr):
    print("Start Dictionary")
    for k,v in arr.items():
        print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x")
        print(k, "   abcdefghijk")
        print(v)
    print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x")
    print("End Dictionary")


def create_rag_qa_chain_with_response_analysis(
    folder_path, 
    embedding_model="mixedbread-ai/mxbai-embed-large-v1", 
    llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct", #"microsoft/phi-2", 
    device="cuda",
    device_num=0
):
    """
    Creates a RetrievalQA chain for analyzing responses based on history, query, and retrieved context.
    
    Parameters:
        folder_path (str): Path to the folder containing PDF files.
        embedding_model (str): Hugging Face embedding model name for embeddings.
        llm_model (str): Hugging Face LLM model name for language model.
        device (str): Device for computation ("cuda" for GPU or "cpu").
        
    Returns:
        Function: A callable chain for response analysis.
    """
    # llm_model="microsoft/phi-2"   
    def cutoff_at_stop_token(response, stop_token="<|end_of_text|>"):
        """
        Truncates the response at the first occurrence of the stop token.
        """
        if isinstance(response, str):
            response = response.strip()
        elif hasattr(response, "content"):
            response = response.content.strip()
        else:
            raise TypeError(f"Unsupported response type: {type(response)}")
        if stop_token in response:
            return response.split(stop_token)[0].strip()
        return response.strip()



    # Step 1: Extract text from PDFs
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

    # Step 2: Split text into chunks
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(text)

    # Step 3: Create a vector store from text chunks
    def get_vectorstore(text_chunks):
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": device})
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # Step 4: Load the LLM
    def get_llm():
        # Initialize the LLM for text generation
        if llm_model == "llama3":
         callbacks = [StreamingStdOutCallbackHandler()]
         llm1 = HuggingFacePipeline.from_model_id(
               model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", #
               task="text-generation",
               device=device_num,  # Use GPU if available
               callbacks=callbacks,  # For streaming outputs
               pipeline_kwargs=dict(
                  return_full_text=False,  # Return only the new tokens
                  max_new_tokens=1024,  # Limit the number of generated tokens
                  do_sample=True,  # Enable sampling for varied outputs
                  temperature=0.5,  # Balance randomness and coherence
                  repetition_penalty = 1.02,
                  min_new_tokens=2,
               ),
               #model_kwargs=dict(load_in_8bit=True) # Add stop tokens here
         )
        elif llm_model == "deepseek":
            callbacks = [StreamingStdOutCallbackHandler()]
            llm1 = HuggingFacePipeline.from_model_id(
                  model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", #
                  task="text-generation",
                  device=device_num,  # Use GPU if available
                  callbacks=callbacks,  # For streaming outputs
                  pipeline_kwargs=dict(
                     return_full_text=False,  # Return only the new tokens
                     max_new_tokens=1024,  # Limit the number of generated tokens
                     do_sample=True,  # Enable sampling for varied outputs
                     temperature=0.5,  # Balance randomness and coherence
                     repetition_penalty = 1.02,
                     min_new_tokens=2,
                  ),
                  #model_kwargs=dict(load_in_8bit=True) # Add stop tokens here
            )
        elif llm_model == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm1 = ChatGoogleGenerativeAI(
                  model="gemini-2.0-flash",
                  disable_streaming=True,
                  # cache=True,
                  temperature=0.5,  # Balance randomness and coherence
                  max_tokens=1024,  # Limit the number of generated tokens
                  timeout=None,
                  max_retries=500,
                  repetition_penalty=1.02,  # Apply repetition penalty
                  min_new_tokens=3,  # Ensure at least 3 tokens are generated
               )
        elif llm_model == "gpt4omini":
            from langchain_openai import OpenAI 
            callbacks = [StreamingStdOutCallbackHandler()]
            llm1 = OpenAI(
            model_name="gpt-4o-mini",
            temperature=0.5,
            max_tokens=1024,
            openai_api_key="",
            streaming=False,  # Optional
            callbacks=callbacks  # Optional
         )
      #   llm1.pipeline.tokenizer.pad_token_id = llm1.pipeline.tokenizer.eos_token_id
        # llm2.pipeline.tokenizer.pad_token_id = llm2.pipeline.tokenizer.eos_token_id
        else:
            raise ValueError(f"Unsupported LLM model: {llm_model}")

        return llm1, llm1

    # Step 5: Create a chain for analyzing responses
    def get_response_analysis_chain(retriever, llm1, llm2):
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
            "Reformulated Statement: What are the advantages of using Chain of Thought reasoning?<|end_of_text|>\n"
            "\n"
            "Example 2\n"
            "Chat History:\n"
            "Human: What is the principle of stare decisis in law?\n"
            "AI: The principle of stare decisis refers to the legal doctrine of adhering to precedent, where courts are bound to follow the decisions of higher courts in similar cases to ensure consistency and predictability in the legal system.\n"
            "Human: Can you explain why it is important in the judicial process?\n"
            "AI: Stare decisis is important because it promotes legal stability, predictability, and fairness. It ensures that similar cases are treated alike, reduces judicial arbitrariness, and helps individuals and businesses understand their legal obligations.\n"
            "Statement: What are some exceptions to this principle?\n"
            "Reformulated Statement: What are some exceptions to the principle of stare decisis?<|end_of_text|>\n"
            "\n"
            "Example 3\n"
            "Chat History:\n"
            "Human: What is a contract?\n"
            "AI: A contract is a legally enforceable agreement between two or more parties that creates mutual obligations. It requires an offer, acceptance, consideration, and the intention to create legal relations to be valid.\n"
            "Human: What happens if one party breaches a contract?\n"
            "AI: If one party breaches a contract, the other party may seek remedies such as damages, specific performance, or rescission, depending on the terms of the contract and the severity of the breach.\n"
            "Statement: Can you explain the difference between damages and specific performance?\n"
            "Reformulated Statement: Can you explain the difference between damages and specific performance as remedies for a contract breach?<|end_of_text|>\n"
            "\n"
            "Example 4\n"
            "Chat History:\n"
            "Human: What is photosynthesis?\n"
            "AI: Photosynthesis is the process by which green plants, algae, and some bacteria convert sunlight, carbon dioxide, and water into glucose and oxygen. It takes place in the chloroplasts of cells and is essential for the survival of most life on Earth.\n"
            "Human: What are the main stages involved?\n"
            "AI: The main stages are the light-dependent reactions, where sunlight is absorbed to produce ATP and NADPH, and the Calvin cycle, where carbon dioxide is fixed into glucose using ATP and NADPH.\n"
            "Statement: What factors can affect this process?\n"
            "Reformulated Statement: What factors can affect the process of photosynthesis?<|end_of_text|>\n"
            "\n"
            "Example 5\n"
            "Chat History:\n"
            "Human: What is intellectual property law?\n"
            "AI: Intellectual property law protects creations of the mind, such as inventions, literary and artistic works, designs, symbols, names, and images. It provides creators exclusive rights to use and benefit from their creations for a certain period.\n"
            "Human: What are the different types of intellectual property?\n"
            "AI: The main types are patents (inventions), copyrights (creative works), trademarks (brand identifiers), and trade secrets (confidential business information).\n"
            "Statement: How do trademarks differ from copyrights?\n"
            "Reformulated Statement: How do trademarks differ from copyrights under intellectual property law?<|end_of_text|>\n"
            "\n"
            "Example 6\n"
            "Chat History:\n"
            "Human: Can you explain what a supply chain is?\n"
            "AI: A supply chain is a network of individuals, organizations, resources, and technologies involved in creating and delivering a product or service, from raw materials to the end customer.\n"
            "Human: What are the key components of a supply chain?\n"
            "AI: Key components include sourcing raw materials, manufacturing, warehousing, transportation, inventory management, and distribution.\n"
            "Statement: What are the main challenges in managing it?\n"
            "Reformulated Statement: What are the main challenges in managing a supply chain?<|end_of_text|>\n"
            "\n"
            "Example 7\n"
            "Chat History:\n"
            "Human: What happens if someone breaches a contract?\n"
            "AI: If someone breaches a contract, remedies are available depending on the circumstances. Common remedies include damages, specific performance, or rescission. To help further, what remedy do you want to pursue in this situation?\n"
            "Statement: Monetary compensation.\n"
            "Reformulated Statement: The remedy the user wants to pursue is monetary compensation.<|end_of_text|>\n"
            "\n"
            "Example 8\n"
            "Chat History:\n"
            "Human: What should I do if my landlord is not making necessary repairs?\n"
            "AI: In cases where a landlord fails to make necessary repairs, you may have legal options such as withholding rent, repairing and deducting costs, or seeking legal action.\n"
            "Human: Are there specific conditions for withholding rent?\n"
            "AI: Yes, withholding rent is typically allowed only after proper notice to the landlord and in jurisdictions where such action is legally permitted. Could you clarify which option you would like to consider?\n"
            "Statement: Withholding rent.\n"
            "Reformulated Statement: The option the user wants to consider is withholding rent.<|end_of_text|>\n"
            "\n"
            "Example 9\n"
            "Chat History:\n"
            "Human: How can I protect my business name from being copied?\n"
            "AI: To protect your business name, you can register it as a trademark, ensure it's unique, and enforce your rights if infringement occurs. Would you like to proceed with trademark registration?\n"
            "Statement: Yes.\n"
            "Reformulated Statement: The user wants to proceed with trademark registration.<|end_of_text|>\n"

            "Based upon the examples do the same task for the following. Ensure to end with <|end_of_text|> token"
            # "You are a question creator. You are given chats of an ai conversation along with a query. Your task is to rewrite it in form of a complete question. You will not answer the query but rather just pass it as it is. In case the query cannot be understood directly but requires context from the conversation, you will pick out the necessary points and add them to the query. At any cost do not answer the query and only state the reformulated query."
            "Chat History: \n {history}\n"
            "Statement: {query}\n"
            "Reformulated Statement:"
        )
        # print(contextualize_q_system_prompt)
        contextualize_q_prompt = PromptTemplate(
            template=contextualize_q_system_prompt,
            input_variables=["history", "query"]
        )
        # Chain for reformulating the query
        reformulate_chain = contextualize_q_prompt | llm1

        memory_prompt = PromptTemplate(
            template=("""
You are an summarizer whose main task is to summarize a chat with special attention to the User. For a given chat history, you need to summarize the conversation in a concise manner. The summary should focus on the key points discussed in the chat and should be user-centric. You should not include any new information or details that were not part of the chat history. The summary should be clear, coherent, and capture the essence of the conversation.
Keep the following points in mind while making the summary:
1. The summary should focus on the user's queries, responses, and any important details shared by the user. It should not miss out any information that the user has provided.
2. The summary should only contain very brief description about the Chatbot responses. It should just state what topics or information the Chatbot has shared without providing specific details
3. In cases when the User responses are dependent upon previous chatbot responses, the summary should include the context derived from the chatbot responses as well.
4. The purpose of the summary should be to provide a clear overview of the chat history that clealy states all information provided by the user during the conversation and the general flow of the conversation
5. If the history only consists of a single general chatbot query then the summary should just mention chatbot asks how it can assists.
6. The summary should always end with an <|end_of_text|> token

Here are a few examples:
Example 1
Input:
Chat History:A
Chatbot: Hi! I am your consumer grievance assistance tool. How can I assist you today?
User: I bought a blender online, and it’s not working properly.
Chatbot: I’m sorry to hear about the issues with your blender. Can you tell me the name of the online store where you made the purchase, as well as any warranty details or receipts you might have?
User: I bought it from HomeMart, but I don’t have the warranty information. What should I do?

Output:
The user reports that they purchased a faulty blender online, which is not working as expected. The chatbot asks the user for more details, specifically the name of the online store and whether the user has any warranty or receipt information. The chatbot offers general advice on how to resolve the issue, mentioning the possibility of contacting the store and checking the warranty policy. The user states that he bought it from HomeMart but does not have the warranty information and asks what to do. <|end_of_text|>

Example 2
Input:
Chat History:
Chatbot: Hi! How can I assist you with your consumer grievance today?
User: I booked a travel package with GoTravel Agency for a vacation, and they’ve delayed everything by a month without informing me.
Chatbot: That sounds incredibly frustrating. Can you confirm if you received any communication from them regarding the delay? Their customer service number is 1800-123-4567, and you may want to contact them directly for clarification.
User: No, they haven’t informed me, and when I called, they said the delay was due to operational issues, but they never explained further. I’ve already paid the full amount!
Chatbot: If you've already paid, the Consumer Protection Act, 2019 provides you with several options. Specifically, Section 2(1)(g) allows you to seek compensation for delays in services. You can demand a refund or request an alternative arrangement within a reasonable time frame.
I can help you draft a sample legal notice to GoTravel Agency for this delay. Here’s an example:
[Your Name]
[Your Address]
[City, State, Pincode]
[Email ID]
[Mobile Number]
[Date]

GoTravel Agency
[Their Address]
[City, State, Pincode]

Subject: Notice for Compensation and Immediate Resolution of Delayed Travel Package

Dear Sir/Madam,

I, [Your Name], booked a vacation package with GoTravel Agency on [Date of Booking], under booking reference number [Booking Reference Number]. The vacation was scheduled for [Original Travel Date], but it has now been delayed by a month, without adequate prior notice or explanation.

I have already paid the full amount of [Amount Paid], and this delay is causing significant inconvenience. Despite contacting your customer service, I have not received any satisfactory resolution or explanation for this delay.

I request that you either provide an alternative vacation package within a reasonable timeframe or issue a full refund of the amount paid. I seek compensation for the inconvenience caused, in the amount of [Compensation Amount], for the breach of the terms of service.

I expect a response within 7 days of receiving this notice. Failure to resolve the matter in this timeframe will compel me to approach the Consumer Forum for legal recourse under the Consumer Protection Act, 2019.

Thank you for your immediate attention to this matter.

Yours faithfully,
[Your Name]

Feel free to customize this notice with your details and send it to GoTravel Agency. If they still don’t respond, you can file a complaint with the National Consumer Disputes Redressal Commission.
User: Thanks for your help. What is the National Consumer Disputes Redressal Commission?

Output:
The user shares that they booked a vacation package with GoTravel Agency, but the company has delayed the vacation by a month without informing the user. The user also mentions paying the full amount upfront. In response, the chatbot sympathizes with the user's frustration and asks whether the user has received any communication from the agency. The chatbot suggests contacting GoTravel Agency's customer service at 1800-123-4567 for clarification. When the user reveals that they were told the delay was due to operational issues, the chatbot explains the user’s rights under the Consumer Protection Act, 2019, specifically Section 2(1)(g), regarding compensation for service delays. The chatbot also offers the option to file a complaint or send a legal notice. At the end, the chatbot provides a sample legal notice for the user to send to GoTravel Agency seeking compensation and resolution. The user thanks the chatbot and asks what the National Consumer Disputes Redressal Commission which the chatbot told he could complain to is. <|end_of_text|>

Example 3
Input:
Chat History:
Chatbot: Hi! How can I assist you with your consumer grievance today?
User: I took my car to QuickFix Auto Services for repair, and it broke down again after just one week. They claim it’s not their fault.
Chatbot: That’s very frustrating! QuickFix Auto Services is obligated to provide quality service under the Consumer Protection Act, 2019, which holds service providers accountable for faulty repairs under Section 2(1)(o). Have you reviewed any repair agreements or warranties they provided?
User: No. Where can I find them?

Output:
The user complains that after taking their car to QuickFix Auto Services for repairs, the car broke down again within a week. The user asks what can be done in this situation. The AI responds by acknowledging the issue and explaining that under the Consumer Protection Act, 2019, the service provider is required to deliver a satisfactory service. The AI also asks if the user has any repair agreements or warranties in place, which could be important for resolving the issue. The conversation ends with the user answering in negative but requesting information about where to get them. <|end_of_text|>


Example 4
Input:
Chat History:
Chatbot: Hi! I am your consumer grievance assistance tool. How can I assist you today?
User: I need help with a faulty product I purchased online.

Output:
The chatbot introduces itself as a consumer grievance assistance tool and asks the user how it can help. The user replies by stating he needs help with a faulty product purchased online.


Now you have to summarize the following chat history. Remember to end with an '<|end_of_text|>' token:
Input:
Chat History:
{history}

Output:
"""),
            input_variables=["history"]
        )
        # Chain for reformulating the query
        memory_chain = memory_prompt | llm1

        # Prompt for the analysis
        analysis_prompt_template = PromptTemplate(
            template=(
                """You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot that provides correct and flawless legal advice to the user. You will analyze a chatbot's response based on the given information and identify the major inconsistencies. The output inconsistencies should be concise and clear, without any repetition. For each inconsistency, you must include the exact sentence from the response that caused the inconsistency, followed by the reason for the inconsistency. Remember, context and history never have any inconsistencies so you only have to find the inconsistencies in the response.

You are provided with:
1. Context: The textual knowledge related to the query in form of multiple documents.
2. History: The previous conversation between the user and the chatbot.
3. Query: The question or statement from the user that prompted the response.
4. Response: The chatbot’s reply to the query only from where the inconsistencies have to be detected.

Evaluation Criteria:
THe chatbot's response will be evaluated based on the following criteria. If any sentence has the following then it is an inconsistency::
1. Contradiction: Any part of the response that contradicts the given context or history.
2. Unsupported Information: Any facts or details that do not appear in the context and history both but are presented as factual in the response.
3. Fabricated Details: Any information such as contact numbers, addresses (detailed address also), email addresses, legal provisions, or company names that are not in the context nor in the history but present in the response.

You are to find inconsistencies only in the Response. No inconsistencies in the history or context or query will be found. Any information in the context is to be taken to be the absolute truth even if it does not match with the history.

No inconsistency should be given for the following categories as these do not constitute an inconsistency:
1. General Knowledge  
   Factual details like contact numbers, email addresses, addresses, websites and web addresses of companies, organizations, or government bodies, or even the customer care numbers are essential for legal discussions and are not to be considered general knowledge. Hence they will be inconsistencies.  
   Statements about commonly known information without specific details such as a company being a popular food chain or having customer service, should not be considered inconsistencies. Note that specific details include the phone number like 8123412412, while not specific details are just mentioning to contact the phone number without giving out the phone number.
   
2. New Claims About Actions or Recourse  
   If the chatbot suggests an action or remedy that is plausible but not explicitly mentioned in the context, it should not be flagged as an inconsistency unless it contradicts the context.

3. Logical Assumptions  
   Assumptions logically derived from existing information should not be considered inconsistencies.  
   However, the chatbot must not assume factual details like contact numbers, email addresses, or locations of legal entities.  
   Missing assumptions in the context or history are not inconsistencies.

4. Irrelevant Details  
   Extra details that do not contradict the context or history should not be considered inconsistencies.  
   Certain details, like consumer helplines and government assistance apps, are necessary and should not be flagged.

5. Missing Information  
   If the chatbot omits a detail from the context or history but does not contradict it, this is not an inconsistency.

6. Partial Information  
   If an inconsistency is flagged because the response provides only part of the context rather than the full information, it should not be considered an inconsistency as long as the given partial information is accurate. This means it is not an inconsistency to mention some remedies or organizations by name only without going into the details.

7. Wrong Behavior  
   If an inconsistency is about how the chatbot should behave or what it should have responded with, it is not an inconsistency as long as there is no contradiction with the context and the history.  
   The evaluator will not judge the chatbot's quality but only whether its responses contradict the given context or history.

8. Notice and Complaint Letter  
   Inconsistencies should not be given for complaint letters and legal notices samples present in the response as even though they might not be part of context, they are just templates and hence should not be considered as inconsistencies. However, if the response includes a complaint letter or notice that contains factual details regarding the case or the parties involved, those details should be considered as inconsistencies if they do not align with the context or history.

9. Repeated Inconsistencies  
   If the same inconsistency is repeated multiple times, whether in exact words or with slight modifications, it should be counted as one inconsistency only, and the excess inconsistencies should be removed.

10. Placeholders
   If the response contains placeholders like [Your Name] or [Your Address], these should not be considered inconsistencies as they are meant to be filled in by the user. However, if the placeholders are used in a way that implies specific information that contradicts the context or history, then it should be flagged as an inconsistency.
---

Output Format:
Based on your evaluation, generate the following structured output:

1. Inconsistencies Present: Yes or No
2. Inconsistencies: [List of inconsistencies. Each inconsistency should include:  
   - Exact sentence from the response  
   - The reason for the inconsistency based on the context. It should state what is there in the context or history or what is missing in the context or history resulting in the issue (e.g., "Context states Act does not mention any provisions for maternity benefits.")
   
   If no inconsistencies, leave it blank.  
]

3. <|end_of_text|>

Remember to end with a <|end_of_text|> token

Example 1
Input:
Context:
Document 0:
The Employment Rights Act 1996 provides employees with certain rights, such as the right to request flexible working, protection against unfair dismissal, and statutory redundancy pay.
Document 1:
The Act does not mention any provisions about maternity benefits or sick pay directly.

History:
The user inquired about employee rights under the Employment Rights Act 1996. The chatbot mentioned rights such as flexible working, protection against unfair dismissal, and statutory redundancy pay. The user also asked if the Act includes maternity benefits, and the chatbot confirmed it does.

Query:
Can you provide more details about maternity benefits under this Act?

Response:
Employment Rights Act 1996 provides certain benefits. Maternity benefits under the Employment Rights Act 1996 include 26 weeks of maternity leave and statutory maternity pay for eligible employees.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. Maternity benefits under the Employment Rights Act 1996 include 26 weeks of maternity leave and statutory maternity pay for eligible employees.
Reason: Context states Act does not mention any provisions for Material benefits. <|end_of_text|>


Example 2
Input:
Context:
Document 0:
Under the Consumer Rights Act 2015, consumers are entitled to a refund, repair, or replacement for goods that are faulty or not as described.
Document 1:
The Act does not include any references to warranty periods being extended for defective goods.

History:
The user asked about extended warranties for faulty products under the Consumer Rights Act 2015. The chatbot confirmed the possibility of an extended warranty and explained that the duration depends on the product type and its condition.

Query:
How long is the extended warranty period?

Response:
The Consumer Rights Act 2015 provides a 12-month extended warranty for defective goods. The Act also provides a way for the warranty periods to be extended for defective goods. The Act also covers refunds and repairs for faulty products.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. The Consumer Rights Act 2015 provides a 12-month extended warranty for defective goods.
Reason: Context states the Act does not include any references to warranty periods being extended for defective goods.
2. The Act also provides a way for the warranty periods to be extended for defective goods.
Reason: Context states The Act does not include any references to warranty periods being extended. <|end_of_text|>

Example 3
Input:
Context:
Document 0:
Section 21 of the Housing Act 1988 allows landlords to evict tenants without providing a reason, as long as the proper notice period is given.
Document 1:
The notice period is currently two months.

History:
The user asked about eviction under Section 21, specifically if a landlord can evict a tenant without giving a reason. The chatbot confirmed this and the user followed up asking about the notice period, to which the chatbot mentioned one month but promised to confirm.

Query:
Can you confirm if one month is the required notice period?

Response:
Yes, the notice period under Section 21 is one month. This has been the case for many years. The landlord is allowed to evict without reason as long as the notice period is followed. The notice must be served in writing and within the guidelines.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. Yes, the notice period under Section 21 is one month.
Reason: Context explicitly states the notice period is two months.<|end_of_text|>


Example 4
Input:
Context:
Document 0:
The Data Protection Act 2018 governs the processing of personal data in the UK. It requires organizations to obtain explicit consent for data processing and outlines individuals' rights regarding access, correction, and deletion of their data.
Document 1:
The Act does not detail penalties for non-compliance.

History:
The user asked for key points of the Data Protection Act 2018. The chatbot explained it governs personal data processing, mandates explicit consent, and provides rights for access, correction, and deletion of data.

Query:
Are there penalties for organizations that fail to comply?

Response:
The Data Protection Act 1995 governs personal data processing and requires explicit consent for such actions. While it outlines individual rights, it does not explicitly detail penalties for non-compliance.

Output:
Output:
Inconsistencies Present: No
Inconsistencies:<|end_of_text|>


Example 5
Input:
Context:
Document 0:
The Equal Pay Act 1970 prohibits discrimination between men and women in terms of pay and employment conditions.
The Act does not specify exceptions or provide details on enforcement mechanisms.

History:
The user asked whether employers can pay men and women differently for the same job. The chatbot referenced the Equal Pay Act 1970, confirming that such discrimination is prohibited.

Query:
Are there any exceptions under this Act?

Response:
The Equal Pay Act 1970 prohibits pay discrimination between men and women for the same job. However, it does allow exceptions for certain job roles based on qualifications and experience. You can also contact +91-9823141241, the contact number of Women's Freedom Organisation for more information. This is their address
Equal Pay Office,
Mumbai Office,
A Wing, 12th Floor, 16/18, J.N. Heredia Marg, Ballard Estate,
Mumbai - 400 001.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. You can also contact +91-9823141241, the contact number of Women's Freedom Organisation for more information.
Reason: Context does not mention a Women’s Freedom Organisation nor its contact number.<|end_of_text|>
2. Air India Limited,
Mumbai Office,
A Wing, 12th Floor, 16/18, J.N. Heredia Marg, Ballard Estate,
Mumbai - 400 001.
Reason: Context does not mention Equal Pay Office or its address.<|end_of_text|>

Based on these examples do the same for the following and remember to always end with an '<|end_of_text|>' token,

Input:
"""



                "Context:\n {context}\nDocument 8: The user can always contact the National Consumer Helpline (1800-11-4000) or UMANG App for immediate assistance or query he or she has. \n"
                "History:\n {history}\n\n"
                "Query:\n {query}\n\n"
                "Response:\n {response}\n\nOutput:"
            ),
            input_variables=["context", "history", "query", "response"]
        )

        analysis_chain = analysis_prompt_template | llm2

        analysis_prompt_template2 = PromptTemplate(
            template=(
                """You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot. You have already identified whether inconsistencies exist in the chatbot’s response. Now, your goal is to analyze the identified inconsistencies, remove the incorrect inconsistencies and assign a Degree of Inconsistency for each of the corrected inconsistencies with the help of the assigned reason. The output should only contain inconsistencies that were originally present in the input and not any new inconsistencies.

Step 1: Removing Incorrect Inconsistencies
An inconsistency must be removed if it falls under any of the following conditions:

1. General Knowledge
- Factual details like contact numbers, email addresses, addresses, websites and webaddresses of companies, organizations, or government bodies or even the customer care numbers are essential for legal discussions and are not to be considered general knowledge. Hence they will be inconsistencies.
- Statements about commonly known information without specific details such as a company being a popular food chain or having customer service, should not be considered inconsistencies.

2. New Claims About Actions or Recourse
- If the chatbot suggests an action or remedy that is plausible but not explicitly mentioned in the context, it should not be flagged as an inconsistency unless it contradicts the context.

3. Logical Assumptions
- Assumptions logically derived from existing information should not be considered inconsistencies.
- However, the chatbot must not assume factual details like contact numbers, email addresses, websites, webaddresses, or locations of legal entities.
- Missing assumptions in the context or conversation history are not inconsistencies.

4. Irrelevant Details
- Extra details that do not contradict the context should not be considered inconsistencies.
- Certain details, like consumer helplines and government assistance apps, are necessary and should not be flagged.

5. Partial Information
- If an inconsistency is flagged because the response provides only part of the context rather than the full information, it should not be considered an inconsistency as long as the given partial information is accurate.

6. Wrong Behavior
- If an inconsistency is about how the chatbot should behave or what it should have responded with, it is not an inconsistency as long as there is no contradiction with the context.
- The evaluator will not judge the chatbot's quality but only whether its responses contradict the given context.

7. Notice and Complaint Letter Samples
- Inconsistencies should not be given for complaint letters and legal notices samples present in the response as even though they might be not be part of context, they are just templates and hence should not be considered as inconsistencies. 
- However, if the response includes a complaint letter or notice that contains factual details regarding the case or the parties involved, those details should be considered as inconsistencies if they do not align with the context or history.

8. Special details
- Consumer can also contact the National Consumer Helpline at 1800-11-4000 or UMANG App for for immediate assistance with their consumer grievance. If none of these options work, you can consider filing a complaint with the District Consumer Commission or the State Consumer Dispute Redressal Commission. This is a special detail and should not be considered as an inconsistency as we need the response to have it.

9. Placeholders
   If the response contains placeholders like [Your Name] or [Your Address], these should not be considered inconsistencies as they are meant to be filled in by the user. However, if the placeholders are used in a way that implies specific information that contradicts the context or history, then it should be flagged as an inconsistency.

Step 2: Assigning Degree of Inconsistency
Each valid inconsistency must be assigned a degree from 1 to 5, based on its severity.

Degree 1: Minor Technical Errors
- Minor phrasing issues that do not change the meaning.
- Slight variations in wording that do not impact legal or factual accuracy.
- Placeholders mean that the user needs to fill in the details and should not be considered as an inconsistency

Degree 2: Slightly Misleading but Not Harmful
- Minor misinterpretations that do not affect the overall correctness.
- Incorrect terminology that does not misguide the user significantly.

Degree 3: Noticeable Errors but Limited Impact
- Providing an incomplete legal explanation while still giving a correct overall direction.
- Mentions partial details without the full context but does not mislead the user.
- Degree 1,2 and 3 imply that no factual information like contact numbers, email addresses, addresses, websites and webaddresses of companies, organizations, or government bodies or even the customer care numbers are present in response but not in the context,

Degree 4: Serious Misleading Information
- Partially incorrect legal or procedural information that could lead to misunderstandings.
- Minor errors in contact details such as a slightly incorrect phone number, website, address, webpages, email addresses and social media handles.
- Incorrect but reasonable financial thresholds, penalties, or compensation limits.
- Contextual misinterpretation leading to a suggestion that is not the best course of action.
- Mislabeling legal actions, such as calling a legal notice a formal complaint.

Degree 5: Critical Errors
- Completely fabricated legal processes, rules, or authorities.
- False or fictional legal remedies or statutory provisions.
- Any information which is not present in the context or history, such as government and organisations names, addresses, phone numbers, emails, or websites. Even if they are considered general knowledge, it will be marked with high degree of inconsistency
- Misdirecting users to the wrong legal body, leading them to file complaints incorrectly.
- Fundamental misrepresentation of laws or legal jurisdictions.
- Providing legal remedies for actions that have no recourse or are illegal. 


Input Provided to You:
1. Inconsistency Present: Yes or No
2. Inconsistencies: [List of identified inconsistencies with their reason]

Output Format:
Based on the inconsistencies provided, generate the following structured output:
Inconsistency Present: Yes or No
Explanation for Correction: [Short and concise explaination for the removals made to the inconsistencies]
Inconsistencies: 
1. Inconsistency: [First Inconsistency]
   Reason: [Reason for the inconsistency as given in Input]
Degree of Inconsistency: [1 to 5]
Explanation: [Short and concise reason for assigning degree based on the corrected inconsistencies]
2. Inconsistency: [Second Inconsistency]
   Reason: [Reason for the inconsistency as given in Input]
Degree of Inconsistency: [1 to 5]
Explanation: [Short and concise reason for assigning degree based on the corrected inconsistencies]
...
<|end_of_text|>

Ensure the output always ends with an '<|end_of_text|>' token and no inconsistency is repeated or added.


Example 1
Input:
Inconsistencies Present: Yes
Inconsistencies:
1. The response states that SpeedCart is a well-known e-commerce platform.
   Reason: The context does not state this.
2. The response provides a tracking number for the user’s order.
   Reason: The context does not contain any tracking information.
3. The response claims that SpeedCart's customer service email is support@speedcart.com.
   Reason: This information is not found in the context.
4. The response states that refunds are processed within 7 days.
   Reason: The context specified 14 days as the refund timeline.
5. Additionally, you can also contact the National Consumer Helpline at 1800-11-4000 for immediate assistance with your consumer grievance.
Reason: Context does not mention the National Consumer Helpline or its contact number.

Output:
Inconsistency Present: Yes
Explanation for Correction: The statement about SpeedCart being a well-known e-commerce platform is general knowledge and not an inconsistency. The tracking number is likely a generated placeholder, not a contradiction. These have been removed.
Inconsistencies:
1. Inconsistency: The response claims that SpeedCart's customer service email is support@speedcart.com.
   Reason: This information is not found in the context.
Degree of Inconsistency: 4
Explanation: Providing an email address not confirmed in the context could mislead users trying to contact customer service.
2. Inconsistency: The response states that refunds are processed within 7 days.
   Reason: The context specified 14 days as the refund timeline.
Degree of Inconsistency: 3
Explanation: The chatbot assumes a specific refund period without any supporting information, which may cause confusion.
3. Inconsistency: Additionally, you can also contact the National Consumer Helpline at 1800-11-4000 for immediate assistance with your consumer grievance.
   Reason: Context does not mention the National Consumer Helpline or its contact number.
Degree of Inconsistency: 1
Explaination: The National Consumer Helpline is a special detail and hence not an inaccuracy.<|end_of_text|>


Example 2
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response states that the Consumer Rights Protection Board resolves all banking disputes.
   Reason: The context does not specify this.
2. The response mentions that users can file complaints at www.consumergrievance.org.
   Reason: This website does not exist.
3. The response provides a compensation range for delayed flights but does not mention that it varies based on the airline’s policy.
   Reason: The context does not specify this.
4. The response claims that the Banking Ombudsman also handles e-commerce disputes.
   Reason: This is not mentioned in the context.

Output:
Inconsistency Present: Yes
Explanation for Correction: The compensation range for delayed flights may be a reasonable assumption rather than a contradiction. This has been removed.
Inconsistencies:
1. Inconsistency: The response states that the Consumer Rights Protection Board resolves all banking disputes.
   Reason: The context does not specify this.
Degree of Inconsistency: 4
Explanation: Misrepresenting the authority of a regulatory body may mislead users into filing complaints in the wrong place.
2. Inconsistency: The response mentions that users can file complaints at www.consumergrievance.org.
   Reason: This website does not exist.
Degree of Inconsistency: 5
Explanation: Providing a non-existent website for filing complaints is a severe error, as it directs users to a fictional resource.
3. Inconsistency: The response claims that the Banking Ombudsman also handles e-commerce disputes.
   Reason: This is not mentioned in the context.
Degree of Inconsistency: 4
Explanation: Assigning incorrect jurisdiction to a regulatory authority may cause users to pursue the wrong legal recourse.<|end_of_text|>


Example 3
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response states that the National Telecom Board resolves disputes related to mobile service providers.
   Reason: This is not mentioned in the context.
2. The response provides an incorrect phone number for the National Telecom Board’s complaint department.
   Reason: This is phone number does not exist in context.
3. The response says that complaints must be filed within 15 days.
   Reason: The context does not specify a timeframe.
4. The response includes steps to file a consumer grievance.
   Reason: The exact process is not stated in the context.
5. The response suggests contacting Airtel customer care number +91-9823141241 for further assistance.
   Reason: This number is not mentioned in the context.


Output:
Inconsistency Present: Yes
Explanation for Correction: The steps for filing a consumer grievance, though not explicitly in the context, are not a contradiction and have been removed.
Inconsistencies:
1. Inconsistency: The response states that the National Telecom Board resolves disputes related to mobile service providers.
   Reason: This is not mentioned in the context.
Degree of Inconsistency: 3
Explanation: Incorrectly attributing dispute resolution authority to an organization can mislead users but does not create direct harm.
2. Inconsistency: The response provides an incorrect phone number for the National Telecom Board’s complaint department.
   Reason: This is phone number does not exist in context.
Degree of Inconsistency: 5
Explanation: Incorrect contact details for an official organization can severely impact users seeking assistance.
3. Inconsistency: The response says that complaints must be filed within 15 days.
   Reason: The context does not specify a timeframe.
Degree of Inconsistency: 3
Explanation: Assigning an arbitrary timeframe without factual support can mislead users about deadlines.
4. Inconsistency: The response suggests contacting Airtel customer care number +91-9823141241 for further assistance.
   Reason: This number is not mentioned in the context.
Degree of Inconsistency: 5
Explanation: Giving wrong factual information like a fake contact number can severely mislead users and obstruct resolution.<|end_of_text|>


Example 4
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response claims that the Food Safety and Standards Authority handles restaurant billing disputes.
   Reason: This is not stated in the context.
2. The response provides a method to escalate complaints through the legal system.
   Reason: The exact process is not given in the context.
3. The response states that users can email complaints to fssaicomplaints@india.org.
   Reason: This email is not given in the context.
4. The response states that restaurant disputes can be reported through the Consumer Protection Board.
   Reason: This is not mentioned in the context while it is a legal concern.
5. The response states to contact the customer service team.
   Reason: This is not mentioned in the context.


Output:
Inconsistency Present: Yes
Explanation for Correction: The general method of escalating complaints through the legal system is not a contradiction and has been removed.
Inconsistencies:
1. Inconsistency: The response claims that the Food Safety and Standards Authority handles restaurant billing disputes.
   Reason: This is not stated in the context.
Degree of Inconsistency: 4
Explanation: Misassigning authority can cause users to file complaints with the wrong regulatory body.
2. Inconsistency: The response states that users can email complaints to fssaicomplaints@india.org.
   Reason: This email is not given in the context.
Degree of Inconsistency: 5
Explanation: Providing a potentially false email address for complaint filing can mislead users and obstruct resolution.
3. Inconsistency: The response states that restaurant disputes can be reported through the Consumer Protection Board.
   Reason: This email is not given in the context.
Degree of Inconsistency: 3
Explanation: While the Consumer Protection Board may be relevant, its role in restaurant disputes is not explicitly mentioned and 4. could be misleading.
4. Inconsistency: The response states to contact the customer service team.
   Reason: This is not mentioned in the context.
Degree of Inconsistency: 1
Explanation: While the contact information is not mentioned in context, the response only states to contact the customer service team without giving the contact information. As there is no misinformation provided, existence of customer service being general knowledge, it has low degree of inconsistency.<|end_of_text|>

Example 5
Input:
Inconsistency Present: No
Inconsistencies:

Output:
Inconsistency Present: No
Explanation for Correction: No inconsistencies present
Inconsistencies:
Degree of Inconsistency: 1
Explanation: No inconsistencies present.<|end_of_text|>


Now based on these examples work for the following and remember to always end with an '<|end_of_text|>' token
Input:
"""
                    


                "{inconsistency}\n\nOutput:"
            ),
            input_variables=["inconsistency"]
        )

        analysis_chain2 = analysis_prompt_template2 | llm2

        # Define a custom chain that combines reformulating and analysis
        class ResponseAnalysisChain:
            def __init__(self, retriever, reformulate_chain, analysis_chain, analysis_chain2, memory_chain):
                self.retriever = retriever
                self.reformulate_chain = reformulate_chain | StrOutputParser()
                self.memory_chain = memory_chain | StrOutputParser()
                self.analysis_chain = analysis_chain | StrOutputParser()
                self.analysis_chain2 = analysis_chain2 | StrOutputParser()


            def __call__(self, inputs):
                # Extract inputs
                start_time_llm = time.time()
                query = inputs["query"]
                history = inputs["history"]
                response = inputs["response"]

                # Reformulate the query
                reformulated_query = self.reformulate_chain.invoke({"history": history, "query": query}) 
                reformulated_query = cutoff_at_stop_token(reformulated_query)

                # Memory for previous history
                memory = self.memory_chain.invoke({"history": history + "\nUser: " + query})
                memory = cutoff_at_stop_token(memory)
                # return memory
            
                # Retrieve relevant context
                context_docs = self.retriever.get_relevant_documents(reformulated_query)
                context = "".join([f"\nDocument {doc_num}:\n{doc.page_content.replace("\n", " ")}" for doc_num, doc in enumerate(context_docs)])

                # Pass everything to the analysis chain
                final_inputs = {
                    "context": context,
                    "chat_history": history,
                    "history": memory,
                    "query": query,
                    "reformulated": reformulated_query,
                    "response": response,
                }
                analysis_temp_ = self.analysis_chain.invoke(final_inputs)
                analysis_temp = cutoff_at_stop_token(analysis_temp_)
                if not any(el.isalpha() for el in analysis_temp):
                    print("EXTREME ERROR:\n", analysis_temp_)
                analysis = cutoff_at_stop_token(self.analysis_chain2.invoke({"inconsistency":analysis_temp}))
                final_inputs['analysis_temp'] = analysis_temp
                final_inputs['analysis'] = analysis
                # final_inputs.pop('history')
                # final_inputs.pop('reformulated')
                disp_dict(final_inputs)
                elapsed = time.time() - start_time_llm
                print("Sleeping for time: ", 24 - elapsed)
                sleep_time_llm = max(0, 24 - elapsed)  # Wait so total time is 15s
                time.sleep(sleep_time_llm)
                return final_inputs

        return ResponseAnalysisChain(retriever, reformulate_chain, analysis_chain, analysis_chain2, memory_chain)

    def detect_hallucination_per_chat(chain, llm):
        # Define a function that performs the analysis dynamically
        def transform_fn(inputs):
            chat = inputs["chat"]  # Expect the input to have a key "chat"
            analysis = []
            analysis_test = []
            outputs = {"turns": {}}
            # print("Inputs:", inputs)
            for i, chat_triple in enumerate(chat, start=1):
                results_dict = chain(chat_triple)
                # print("Result:", results_dict)
                result = results_dict["analysis"]
                if(result.strip().startswith("Inconsistencies Present: No") or result.strip().startswith("Inconsistency Present: No")):
                    print("No inconsistencies found in turn ", i)
                    results_dict["modified_analysis"] = ""
                    outputs["turns"][i] = copy.deepcopy(results_dict)
                    continue
                analysis_test.append("\nTurn "+str(i)+ ":\n"+result)
                # Extract the degree of inconsistency from the result (assuming it's explicitly stated in the result text).
                analysis_test.append(result)
                # degree_line = [line for line in result.split("\n") if "Degree of Inconsistency:" in line]
                # if degree_line:
                #     degree = int(degree_line[0].split(":")[1].strip())
                #     result_lines = result.split("\n")
                #     line_no = 0
                #     for line_no in range(len(result_lines)):
                #         if "Explanation for Correction:" in result_lines[line_no]:
                #             break
                #     inconsistencies_text = "\n".join(result_lines[:line_no]) 
                #     if degree >= 4:  # Include only if degree is 4 or 5
                #         analysis.append("\nTurn " + str(i + 1) + "\n" + inconsistencies_text)
                pattern = re.compile(r'Inconsistency:(.*?)Degree of Inconsistency: (\d+)', re.DOTALL)
                matches = pattern.findall(str(result))
                # print("PAttern, MATCH")
                # print(pattern, matches)
                analysis_turn = ""
                found = False
                for j, (match, degree) in enumerate(matches, start=1):
                    if int(degree) >= 4:
                        if(not found):
                            analysis_turn = f"Turn {i}\nInconsistencies Present: Yes\nInconsistencies:\n"
                            found = True
                        analysis_turn = analysis_turn + f"{j}. {match.strip()} Degree of Inconsistency is {degree}\n"
                if found==True:
                    analysis.append(analysis_turn)
                results_dict["modified_analysis"] = analysis_turn
                outputs["turns"][i] = copy.deepcopy(results_dict)
            if len(analysis)==0:
                analysis.append("Turn 1\nInconsistencies Present: No\nTurn 2\nInconsistencies Present: No\n")
            analysis_text = "\n".join(analysis)
            # analysis_text = "\n".join(analysis_test)
    
            # Save the analysis_test variable for further analysis
            # with open('./hall_detect/exp_test_files/analysis_test_results.txt', 'w') as f:
            #     for item in analysis_test:
            #         f.write(f"{item}<|end_of_element|>")
            # Create the analysis prompt
            prompt = PromptTemplate(
                template=(
                    # "You are given an individual history of Inconsistencies detected in a chatbot. You are an evaluator and your task is to use all of these statements and give out simple description answering whether Inconsistencies are present as a whole in the chatbot. Do not individually mention each Inconsistencies but give a single statement describing whether Inconsistencies are present or not. It should be in this format:"
                    # "Inconsistencies are present"
                    # "The folllowing information is inconsistent"
                    # "\nAnswer in short. What is the final response to the questions: Are there any Inconsistencies in this chat? List the most significant of them"
"""You are an evaluator tasked with analyzing inconsistencies in a chatbot’s responses. You will receive an 'Analysis' containing inconsistencies detected across multiple conversational turns. Your goal is to generate a unified, concise list of inconsistencies covering the entire conversation while eliminating redundancy. Follow these guidelines:

1. Ensure that no inconsistency is missed and that no extra inconsistency is added.  
2. The inconsistencies should be clear, direct, and not overly verbose.  
3. Make sure to capture the exact words used in the inconsistencies.  
4. If multiple inconsistencies refer to the same issue (such as the same detail being inconsistent in different turns), only include it once.  
5. Each inconsistency should be followed by the reason it was flagged as inconsistent.  
6. Once the output is finished, end with the <|end_of_text|>token.  

Input Format:  
- A list of conversational turns, each containing detected inconsistencies (with reasons).  
- Each turn may or may not have inconsistencies present.  
- The inconsistencies are listed for each turn separately.  

Output Format:  
- If inconsistencies are detected:  
  Inconsistencies detected: Yes.  
  The following information is inconsistent:  
  1. [First inconsistency]  
     Reason: [Reason for inconsistency]  
  2. [Second inconsistency]  
     Reason: [Reason for inconsistency]  
  3. [Third inconsistency]  
     Reason: [Reason for inconsistency]  
  ...  
  <|end_of_text|> 

- If no inconsistencies are detected:  
  Inconsistencies detected: No.  
  <|end_of_text|> 

Examples:  

Example 1  

Input:  
Turn 2  
Inconsistencies Present: Yes  
Inconsistencies:  
1. In case of any doubts +91-9931223120 is the phone number for customer support.  
   Reason: The context does not provide a phone number for customer support.  

Turn 4  
Inconsistencies Present: Yes  
Inconsistencies:  
1. You can visit the government portal at www.example.gov.in for more details.  
   Reason: The context does not mention any government portal or provide a link.  
2. The refund process takes 30 days to complete.  
   Reason: The context does not specify the duration of the refund process.  

Output:  
Inconsistencies detected: Yes.  
The following information is inconsistent:  
1. In case of any doubts +91-9931223120 is the phone number for customer support.  
   Reason: The context does not provide a phone number for customer support.  
2. You can visit the government portal at www.example.gov.in for more details.  
   Reason: The context does not mention any government portal or provide a link.  
3. The refund process takes 30 days to complete.  
   Reason: The context does not specify the duration of the refund process.  
<|end_of_text|> 

Example 2  

Input:  
Turn 3  
Inconsistencies Present: Yes  
Inconsistencies:  
1. You can take legal action immediately.  
   Reason: The context does not specify the timeframe for taking legal action.  

Turn 5  
Inconsistencies Present: Yes  
Inconsistencies:  
1. You can file a consumer complaint via email at support@company.com.  
   Reason: The context does not provide an email filing option for complaints.  

Output:  
Inconsistencies detected: Yes.  
The following information is inconsistent:  
1. You can take legal action immediately.  
   Reason: The context does not specify the timeframe for taking legal action.  
2. You can file a consumer complaint via email at support@company.com.  
   Reason: The context does not provide an email filing option for complaints.  
<|end_of_text|> 

Example 3  

Input:  
Turn 1  
Inconsistencies Present: No  

Output:  
Inconsistencies detected: No.  
<|end_of_text|> 
"""

                    "Based upon the examples, do the same with the following"
                    "Input:"
                    "{analysis}\n\n"
                    "Output:"
                ),
                input_variables=["analysis"]
            )
            # Use the prompt and LLM in a chain
            combination_chain = prompt | llm

            #print("Analysis Total: abcdefg\n")
            resul = cutoff_at_stop_token(combination_chain.invoke({"analysis": analysis_text}))

            outputs["analysis_combined"] = analysis_text
            outputs["result"] = resul
            # print("OUTPUTS   abcdefg\n")
            # print(outputs)
            # if len(outputs["turns"])==0:
            #     print("ERROR TURN IS MISSING\n")
            #     print("Analysis Total: abcdefg\n", analysis)
            #     print("Analysis Test: abcdefg\n", analysis_test)
            return outputs
            # return {"analysis": analysis_text,"result": }
        # Return a chain wrapping the transformation function
        return TransformChain(input_variables=["chat"], output_variables=["result"], transform=transform_fn)

    # Build the pipeline
    text = get_pdf_text(folder_path)
    text_chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm1, llm2 = get_llm()
    response_analysis_chain = get_response_analysis_chain(retriever, llm1, llm2)
    final_chain = detect_hallucination_per_chat(response_analysis_chain, llm1)

    return final_chain

import re
def split_chat_into_elements(chat_sequence):
    """
    Splits the chat sequence into a list of elements where each element
    is a complete "AI:" or "Human:" message.
    
    Parameters:
        chat_sequence (str): A single string containing the entire chat sequence.

    Returns:
        list of str: A list of messages, each starting with "AI:" or "Human:".
    """
    # Split the chat sequence into segments using "Human:" and "AI:" as boundaries
    elements = re.split(r"(?<=\n)(?=Human:|AI:)", chat_sequence.strip())
    return [element.strip().replace("Human:", "User:").replace("AI:", "Chatbot:") for element in elements if element.strip()]


def build_history_query_pairs(elements):
    """
    Recursively creates history-query pairs from the list of elements.

    Parameters:
        elements (list of str): List of chat elements, alternating between "AI:" and "Human:".

    Returns:
        list of dict: A list where each entry contains:
            - "history": The history up to the current query.
            - "query": The latest user input.
    """
    pairs = []

    for i in range(len(elements)):
        # Only process when a Human message is found
        if elements[i].startswith("Chatbot:") and i>0:
            query = elements[i-1].replace("User:", "").strip()
            history = "\n".join(elements[:i-1]).strip()
            response = elements[i].replace("Chatbot:", "").strip()
            pairs.append({
                "history": history,
                "query": query,
                "response": response
            })

    return pairs

# Wrapper function to handle chat sequence
def process_chat_sequence(chat_sequence):
    """
    Processes the chat sequence to generate history-query pairs.

    Parameters:
        chat_sequence (str): A single string containing the entire chat sequence.

    Returns:
        list of dict: A list of history-query pairs.
    """
    elements = split_chat_into_elements(chat_sequence)
    return build_history_query_pairs(elements)


folder_path = "./hall_detect/rag"

ver = "1"
# Example query
data = pd.read_csv("./chatbot/chat_generation/results2/combined_new.csv")
data.rename(columns={"Simulated Chat":"Chat"}, inplace=True)
data.dropna(inplace=True, subset = ["Chat"])
data.reset_index(inplace=True, drop=True)
chats = data["Chat"].to_list()
chatbot_version = data["Chatbot_version"].to_list()
folder = "deepseek" 
folder = "gemini"
# folder = "gpt4omini"
# REMEMBER TO REMOVE SLEEP LOGIC IF NOT USING GEMINI
response_analysis_chain = create_rag_qa_chain_with_response_analysis(folder_path, llm_model=folder, device=1)
output_file = "./chatbot/chat_generation/critic3_v10_" + folder + "/file_output_v" +ver +".txt"
final_df_loc = "./chatbot/chat_generation/critic3_v10_" + folder + "/result_df_v" +ver +".csv"
csv_file = "./chatbot/chat_generation/critic3_v10_" + folder + "/output_df_v" +ver +".csv"
final_var = "./chatbot/chat_generation/critic3_v10_" + folder + "/output_df_v" +ver +".pt"


with open(output_file, "a") as f:
    f.write("Start\n\n\n\n\n\n")  
result = []
try:
    df_final = pd.read_csv(final_df_loc, index_col=0)
    print("Starting from ", len(df_final))
except:
    df_final = pd.DataFrame(columns=["Chat", "Turns", "Result", "Version"])
    print("Starting from 0")
df_final_list = df_final.to_dict("records")
for chat_no, (chat, ver) in enumerate(zip(chats[len(df_final):], chatbot_version[len(df_final):]), start=len(df_final)):
    print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x")
    print("START CHAT " + str(chat_no))
    res = {'analysis': "Sample analysis text", 'result': "Sample result text", 'logs': "Sample logs text"}
    res = response_analysis_chain.invoke({"chat": process_chat_sequence(chat)})
    disp_dict({'Analysis':res['analysis_combined'], 'Result':res['result']})
    result.append(res['result'])
    df_final_list.append({"Chat": chat, "Turns": res["turns"], "Analysis_Combined":res["analysis_combined"], "Result": res['result'], "Version":ver})
    df_final = pd.DataFrame(df_final_list)
    df_final.to_csv(final_df_loc)
    
    with open(output_file, "a") as f:
            f.write("Chat:-\n" + chat + "\nResult:-" + res['result'] + "\n\n\n\n\n\n")  # Separate each string by a blank line
    print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x")

for r in result:
    print(r, "1234")

# output_file = "./hall_detect/critic3_v5/output2_" +ver +".txt"
# with open(output_file, "a") as f:
#     for string in result:
#         f.write(string + "\n\n\n\n\n\n")  # Separate each string by a blank line

data["result"] = result  # Replace 'column_name' with your desired column name
df = pd.DataFrame(data)


df.to_csv(csv_file)

end_time = datetime.now()
print("End Time: ", end_time)
print("Time Taken: ", (end_time-start_time))


# n/legalLLM/chatbot/chat_generation/critic3_v10.py > ./chatbot/chat_generation/critic3_v10_deepseek/std_output_v1.txt 2>&1
# [1] 2051264
# (legalllm) (base) common_user@iitb-dgx6:~/spandan/legalLLM$ nohup /raid/nlp/common_user/miniconda3/envs/legalllm/bin/python3 /raid/nlp/common_user/spandan/legalLLM/chatbot/chat_generation/critic3_v10.py > ./chatbot/chat_generation/critic3_v10_gpt4omini/std_output_v1.txt 2>&1 &
# [2] 2167141
# bash: ./chatbot/chat_generation/critic3_v10_gpt4omini/std_output_v1.txt: No such file or directory
# [2]+  Exit 1                  nohup /raid/nlp/common_user/miniconda3/envs/legalllm/bin/python3 /raid/nlp/common_user/spandan/legalLLM/chatbot/chat_generation/critic3_v10.py > ./chatbot/chat_generation/critic3_v10_gpt4omini/std_output_v1.txt 2>&1
# (legalllm) (base) common_user@iitb-dgx6:~/spandan/legalLLM$ nohup /raid/nlp/common_user/miniconda3/envs/legalllm/bin/python3 /raid/nlp/common_user/spandan/legalLLM/chatbot/chat_generation/critic3_v10.py > ./chatbot/chat_generation/critic3_v10_gpt4omini/std_output_v1.txt 2>&1 &
# [2] 2168069
