import os
from PyPDF2 import PdfReader
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.transform import TransformChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_huggingface import HuggingFacePipelinex
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
    device="cuda"
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
        callbacks = [StreamingStdOutCallbackHandler()]
        llm1 = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            task="text-generation",
            device=0,  # Use GPU if available
            callbacks=callbacks,  # For streaming outputs
            pipeline_kwargs=dict(
                return_full_text=False,  # Return only the new tokens
                max_new_tokens=1024,  # Limit the number of generated tokens
                do_sample=True,  # Enable sampling for varied outputs
                temperature=0.5,  # Balance randomness and coherence
                repetition_penalty = 1.02,
            ),
            #model_kwargs=dict(load_in_8bit=True) # Add stop tokens here
        )
        # llm2 = HuggingFacePipeline.from_model_id(
        #     model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        #     task="text-generation",
        #     device=1,  # Use GPU if available
        #     callbacks=callbacks,  # For streaming outputs
        #     pipeline_kwargs=dict(
        #         return_full_text=True,  # Return only the new tokens
        #         max_new_tokens=1024,  # Limit the number of generated tokens
        #         do_sample=True,  # Enable sampling for varied outputs
        #         temperature=0.7,  # Balance randomness and coherence
        #     ),
        # )

        # Set pad_token_id to eos_token_id for proper padding
        llm1.pipeline.tokenizer.pad_token_id = llm1.pipeline.tokenizer.eos_token_id
        # llm2.pipeline.tokenizer.pad_token_id = llm2.pipeline.tokenizer.eos_token_id

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
                """You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot that provides correct and flawless legal advice to the user. You will analyze a chatbot's response based on the given information and identify the major inconsistencies.

You are provided with:
- Context: The textual knowledge related to the query in form of multiple documents.
- History: The previous conversation between the user and the chatbot.
- Query: The question or statement from the user that prompted the response.
- Response: The chatbot’s reply to the query.

Evaluation Criteria:
You must determine whether the chatbot’s response contains inconsistencies based on the following:
1. Contradiction: Any part of the response that contradicts the given context and history.
2. Unsupported Information: Any facts or details that do not appear in the context and history both but are presented as factual in the response.
3. Fabricated Details: Any made-up information such as contact numbers, addresses, legal provisions, or company names that are not in the context and history both but present in the response.

You are to find inconsistencies only in the Response. No inconsistencies in the history or context or query will be found. Any information in the context is to be taken to be the absolute truth even if is does not match with the history.

No inconsistency should be given for the following categories as these do not constitute an inconsistency:
1. General Knowledge
- Statements about widely known facts, such as a company being a popular food chain or having customer service, should not be considered inconsistencies.
- The focus is on legal and consumer laws, not general knowledge.
- However, factual details like contact numbers, emails, addresses, and websites of companies, organizations, or government bodies are essential for legal discussions and should remain as inconsistencies.

2. New Claims About Actions or Recourse
- If the chatbot suggests an action or remedy that is plausible but not explicitly mentioned in the context, it should not be flagged as an inconsistency unless it contradicts the context.

3. Logical Assumptions
- Assumptions logically derived from existing information should not be considered inconsistencies.
- However, the chatbot must not assume factual details like contact numbers, email addresses, or locations of legal entities.
- Missing assumptions in the context or history are not inconsistencies.

4. Irrelevant Details
- Extra details that do not contradict the context or history should not be considered inconsistencies.
- Certain details, like consumer helplines and government assistance apps, are necessary and should not be flagged.

5. Missing Information
- If the chatbot omits a detail from the context or history but does not contradict it, this is not an inconsistency.

6. Partial Information
- If an inconsistency is flagged because the response provides only part of the context rather than the full information, it should not be considered an inconsistency as long as the given partial information is accurate. This means it is not an inconsistency to mention some remedies or organisations by name only without going into the details.

7. Wrong Behavior
- If an inconsistency is about how the chatbot should behave or what it should have responded with, it is not an inconsistency as long as there is no contradiction with the context and the history.
- The evaluator will not judge the chatbot's quality but only whether its responses contradict the given context or history.

8. Notice and Complaint Letter
- Inconsistenies should not be given for complaint letters and legal notices samples present in the response as even though they might be not be part of context, they are just templates and hence should not be considered as inconsistencies. However if the response includes a complaint letter or notice that contains factual details regarding the case or the parties involved, those details should be considered as inconsistencies if they do not allign with the context or history.

9. Repeated Inconsistencies
- If same inconsistency is repeated multiple times whether in exact words or with slight modifications, it should be counted as one inconsistency only and the excess inconsistencies should be removed.

Output Format:
Based on your evaluation, generate the following structured output:
1. Inconsistency Present: Yes or No
2. Inconsistencies: [List of inconsistencies without repetition only if Yes in Inconsistency Present. If No, leave it blank. The list of inconsistency should not belong to the above mentioned categories]
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
Maternity benefits under the Employment Rights Act 1996 include 26 weeks of maternity leave and statutory maternity pay for eligible employees.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. The response states that the Employment Rights Act 1996 includes maternity benefits, but the context explicitly states that it does not mention them.
2. The details about “26 weeks of maternity leave” and “statutory maternity pay” are not supported by the context.<|end_of_text|>


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
The Consumer Rights Act 2015 provides a 12-month extended warranty for defective goods.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. The response falsely claims that the Act provides for an extended warranty period, which is explicitly not mentioned in the context.
2. The "12-month extended warranty" detail is fabricated and unsupported by the context.<|end_of_text|>


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
Yes, the notice period under Section 21 is one month.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. The response incorrectly states that the notice period under Section 21 is one month, while the context explicitly states it is two months.<|end_of_text|>


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
The Equal Pay Act 1970 prohibits pay discrimination between men and women for the same job. You can also contact +91-9823141241, the contact number of Women's Freedom Organisation for more information.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. The response introduces a Women's Freedom Organisation and its phone number, neither of which is present in the context.<|end_of_text|>


Based on these examples do the same for the following and remember to always end with an '<|end_of_text|>' token,

Input:
"""



                "Context:\n {context}\n"
                "History:\n {history}\n\n"
                "Query:\n {query}\n\n"
                "Response:\n {response}\n\nOutput:"
            ),
            input_variables=["context", "history", "query", "response"]
        )

        analysis_chain = analysis_prompt_template | llm2

        analysis_prompt_template2 = PromptTemplate(
            template=(
                """You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot. You have already identified whether inconsistencies exist in the chatbot’s response. Now, your goal is to analyze the identified inconsistencies, remove the wrong inconsistencies and assign a Degree of Inconsistency for each of the corrected inconsistencies. The output should only contain inconsistencies that were originally present in the input and not any new inconsistencies.

Task 1: Removing Incorrect Inconsistencies
An inconsistency must be removed if it falls under any of the following conditions:

1. General Knowledge
- Statements about widely known facts, such as a company being a popular food chain or having customer service, should not be considered inconsistencies.
- The focus is on legal and consumer laws, not general knowledge.
- However, factual details like contact numbers, email addresses, addresses, websites and webaddresses of companies, organizations, or government bodies are essential for legal discussions and should remain as inconsistencies.

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
- Consumer can also contact the National Consumer Helpline at 1800-11-4000 or UMANG App for for immediate assistance with their consumer grievance. This is a special detail and should not be considered as an inconsistency as we need the response to contain it.

Task 2: Assigning Degree of Inconsistency
Each valid inconsistency must be assigned a degree from 1 to 5, based on its severity.

Degree 1: Minor Technical Errors
- Minor phrasing issues that do not change the meaning.
- Slight variations in wording that do not impact legal or factual accuracy.

Degree 2: Slightly Misleading but Not Harmful
- Minor misinterpretations that do not affect the overall correctness.
- Incorrect terminology that does not misguide the user significantly.

Degree 3: Noticeable Errors but Limited Impact
- Errors in minor procedural details that do not prevent the user from taking correct action.
- Providing an incomplete legal explanation while still giving a correct overall direction.
- Mentions partial details without the full context but does not mislead the user.

Degree 4: Serious Misleading Information
- Partially incorrect legal or procedural information that could lead to misunderstandings.
- Minor errors in contact details such as a slightly incorrect phone number, website, address, webadrreses, email addresses and social media handles.
- Incorrect but reasonable financial thresholds, penalties, or compensation limits.
- Contextual misinterpretation leading to a suggestion that is not the best course of action.
- Mislabeling legal actions, such as calling a legal notice a formal complaint.

Degree 5: Critical Errors
- Completely fabricated legal processes, rules, or authorities.
- False or fictional legal remedies or statutory provisions.
- Incorrect or fictional contact information, such as fake government and organisations, office addresses, phone numbers, emails, or websites.
- Misdirecting users to the wrong legal body, leading them to file complaints incorrectly.
- Fundamental misrepresentation of laws or legal jurisdictions.
- Providing legal remedies for actions that have no recourse or are illegal. 


Input Provided to You:
1. Inconsistency Present: Yes or No
2. Inconsistencies: [List of identified inconsistencies]

Output Format:
Based on the inconsistencies provided, generate the following structured output:
Inconsistency Present: Yes or No
Explanation for Correction: [Short and concise reason for the corrections made to the inconsistencies]
Inconsistencies: 
Inconsistency: [First Inconsistency]
Degree of Inconsistency: [1 to 5]
Explanation: [Short and concise reason for assigning degree based on the corrected inconsistencies]
Inconsistency: [Second Inconsistency]
Degree of Inconsistency: [1 to 5]
Explanation: [Short and concise reason for assigning degree based on the corrected inconsistencies]
...
<|end_of_text|>

Ensure the output always ends with an '<|end_of_text|>' token and no inconsistency is repeated


Example 1
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response states that SpeedCart is a well-known e-commerce platform, but the context does not mention this.
2. The response provides a tracking number for the user’s order, but the context does not contain any tracking information.
3. The response claims that SpeedCart's customer service email is support@speedcart.com, but this information is not found in the context.
4. The response states that refunds are processed within 7 days, but the context does not specify a refund timeline.

Output:
Inconsistency Present: Yes
Explanation for Correction: The statement about SpeedCart being a well-known e-commerce platform is general knowledge and not an inconsistency. The tracking number is likely a generated placeholder, not a contradiction. These have been removed.
Inconsistencies:
1. Inconsistency: The response claims that SpeedCart's customer service email is support@speedcart.com, but this information is not found in the context.
Degree of Inconsistency: 4
Explanation: Providing an unsupported email address could mislead users trying to contact customer service.
2. Inconsistency: The response states that refunds are processed within 7 days, but the context does not specify a refund timeline.
Degree of Inconsistency: 3
Explanation: The chatbot assumes a specific refund period without any supporting information, which may cause confusion.<|end_of_text|>


Example 2
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response states that the Consumer Rights Protection Board resolves all banking disputes, but the context does not specify this.
2. The response mentions that users can file complaints at www.consumergrievance.org, but this website does not exist.
3. The response provides a compensation range for delayed flights but does not mention that it varies based on the airline’s policy.
4. The response claims that the Banking Ombudsman also handles e-commerce disputes, which is not mentioned in the context.

Output:
Inconsistency Present: Yes
Explanation for Correction: The compensation range for delayed flights may be a reasonable assumption rather than a contradiction. This has been removed.
Inconsistencies:
1. Inconsistency: The response states that the Consumer Rights Protection Board resolves all banking disputes, but the context does not specify this.
Degree of Inconsistency: 4
Explanation: Misrepresenting the authority of a regulatory body may mislead users into filing complaints in the wrong place.
2. Inconsistency: The response mentions that users can file complaints at www.consumergrievance.org, but this website does not exist.
Degree of Inconsistency: 5
Explanation: Providing a non-existent website for filing complaints is a severe error, as it directs users to a fictional resource.
3. Inconsistency: The response claims that the Banking Ombudsman also handles e-commerce disputes, which is not mentioned in the context.
Degree of Inconsistency: 4
Explanation: Assigning incorrect jurisdiction to a regulatory authority may cause users to pursue the wrong legal recourse.<|end_of_text|>


Example 3
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response states that the National Telecom Board resolves disputes related to mobile service providers, but this is not mentioned in the context.
2. The response provides an incorrect phone number for the National Telecom Board’s complaint department.
3. The response says that complaints must be filed within 15 days, but the context does not specify a timeframe.
4. The response includes steps to file a consumer grievance, but the exact process is not stated in the context.
5. The response suggest contacting Airtel customer care number +91-9823141241 for further assistance, but this number is not mentioned in the context.

Output:
Inconsistency Present: Yes
Explanation for Correction: The steps for filing a consumer grievance, though not explicitly in the context, are not a contradiction and have been removed.
Inconsistencies:
1. Inconsistency: The response states that the National Telecom Board resolves disputes related to mobile service providers, but this is not mentioned in the context.
Degree of Inconsistency: 3
Explanation: Incorrectly attributing dispute resolution authority to an organization can mislead users but does not create direct harm.
2. Inconsistency: The response provides an incorrect phone number for the National Telecom Board’s complaint department.
Degree of Inconsistency: 5
Explanation: Incorrect contact details for an official organization can severely impact users seeking assistance.
3. Inconsistency: The response says that complaints must be filed within 15 days, but the context does not specify a timeframe.
Degree of Inconsistency: 3
Explanation: Assigning an arbitrary timeframe without factual support can mislead users about deadlines.
4. Inconsistency: The response suggest contacting Airtel customer care number +91-9823141241 for further assistance, but this number is not mentioned in the context.
Degree of Inconsistency: 5
Explanation: Giving wrong factual information like a fake contact number can severely mislead users and obstruct resolution.<|end_of_text|>


Example 4
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response claims that the Food Safety and Standards Authority handles restaurant billing disputes, but this is not stated in the context.
2. The response provides a method to escalate complaints through the legal system, but the exact process is not given in the context.
3. The response states that users can email complaints to fssaicomplaints@india.org, but this email is not verified.
4. The response states that restaurant disputes can be reported through the Consumer Protection Board, but this is not mentioned in the context.
5. The response states to contact the customer service team, which is not mentioned in context.

Output:
Inconsistency Present: Yes
Explanation for Correction: The general method of escalating complaints through the legal system is not a contradiction and has been removed.
Inconsistencies:
1. Inconsistency: The response claims that the Food Safety and Standards Authority handles restaurant billing disputes, but this is not stated in the context.
Degree of Inconsistency: 4
Explanation: Misassigning authority can cause users to file complaints with the wrong regulatory body.
2. Inconsistency: The response states that users can email complaints to fssaicomplaints@india.org, but this email is not verified.
Degree of Inconsistency: 5
Explanation: Providing a potentially false email address for complaint filing can mislead users and obstruct resolution.
3. Inconsistency: The response states that restaurant disputes can be reported through the Consumer Protection Board, but this is not mentioned in the context.
Degree of Inconsistency: 3
Explanation: While the Consumer Protection Board may be relevant, its role in restaurant disputes is not explicitly mentioned and could be misleading.<|end_of_text|>
4. Inconsistency: The response states to contact the customer service team, which is not mentioned in context.
Degree of Inconsistency: 1
Explanation: While the contact information is not mentioned in context, the response only states to contact the customer service team without giving the contact information. As there is no misinformation provided, existence of customer service being general knowledge, it has low degree of inconsistency<|end_of_text|>

Example 5
Input:
Inconsistency Present: No
Inconsistencies:

Output:
Inconsistency Present: No
Explanation for Correction: No inconsistencies present
Inconsistencies:
Degree of Inconsistency: 1
Explanation: No inconsistencies present<|end_of_text|>


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
                    "history": memory,
                    "query": query,
                    "reformulated": reformulated_query,
                    "response": response,
                }
                analysis_temp = cutoff_at_stop_token(self.analysis_chain.invoke(final_inputs))
                analysis = cutoff_at_stop_token(self.analysis_chain2.invoke({"inconsistency":analysis_temp}))
                final_inputs['analysis_temp'] = analysis_temp
                final_inputs['analysis'] = analysis
                # final_inputs.pop('history')
                # final_inputs.pop('reformulated')
                disp_dict(final_inputs)
                global first
                if first==True:
                    # print(final_inputs['context'].replace('/n', ' '))
                    first = False
                return analysis

        return ResponseAnalysisChain(retriever, reformulate_chain, analysis_chain, analysis_chain2, memory_chain)

    def detect_hallucination_per_chat(chain, llm):
        # Define a function that performs the analysis dynamically
        def transform_fn(inputs):
            chat = inputs["chat"]  # Expect the input to have a key "chat"
            analysis = []
            analysis_test = []
            for i, chat_triple in enumerate(chat, start=1):
                result = chain(chat_triple)
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

Ensure no inconsistency is missed and no extra inconsistency is added.
Merge similar inconsistencies into a single, concise statement.
The inconsistencies should be clear, direct, and not overly verbose.
Do not include the degree of inconsistency in the output—only the inconsistencies themselves.
Avoid repetition by ensuring each inconsistency is listed only once, even if it appears multiple times in different turns.
As soon as the output is finished, end with the <|end_of_text|> token.

Input Format:
- A list of conversational turns, each containing detected inconsistencies.
- Each turn may or may not have inconsistencies present.
- The inconsistencies are listed for each turn separately.

Output Format:
- If inconsistencies are detected:
Inconsistencies detected: Yes. 
The following information is inconsistent: 
1. [First inconsistency]
2. [Second inconsistency]
3. [Third inconsistency]<|end_of_text|>

- If no inconsistencies are detected:
Inconsistencies detected: No.<|end_of_text|>

Here are a few examples:

Example 1:
Input:
Analysis:
Turn 2
Inconsistencies Present: Yes
Inconsistencies:
1. The response provides a phone number for customer support, but this is not mentioned in the context.

Turn 4
Inconsistencies Present: Yes
Inconsistencies:
1. The response includes a link to a government portal that is not referenced in the context.
2. The response suggests that the refund process takes 30 days, which is not stated in the context.

Output: 
Inconsistencies detected: Yes.
The following information is inconsistent:
1. The response provides a phone number for customer support that is not mentioned in the context.
2. The response includes a link to a government portal that is not referenced in the context.
3. The response suggests that the refund process takes 30 days, which is not supported by the context.<|end_of_text|>


Example 2:
Input:
Analysis:
Turn 1
Inconsistencies Present: No

Turn 3
Inconsistencies Present: Yes
Inconsistencies:
1. The response states that legal action can be taken immediately, but the context does not confirm this timeframe.

Turn 5
Inconsistencies Present: Yes
Inconsistencies:
1. The response claims that a consumer complaint can be filed via email, but no such method is mentioned in the context.

Output:
Inconsistencies detected: Yes.
The following information is inconsistent:
1. The response states that legal action can be taken immediately, but the context does not confirm this timeframe.
2. The response claims that a consumer complaint can be filed via email, but no such method is mentioned in the context.<|end_of_text|>


Example 3:
Input:
Analysis:
Turn 1
Inconsistencies Present: No

Output: 
Inconsistencies detected: No.<|end_of_text|>
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
            return {"analysis": analysis_text,"result": cutoff_at_stop_token(combination_chain.invoke({"analysis": analysis_text}))}
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

# Create the RAG-based QA chain
import time
start_time = time.time()
print("Start Time: ", start_time)
print("START")
# Example query
first = True
folder_path = "./hall_detect/rag"
response_analysis_chain = create_rag_qa_chain_with_response_analysis(folder_path)
chats = [
'''AI: Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you.


Human: I need gas


AI: You're looking for information related to gas, specifically LPG. You can consider it as a consumer grievance. I'm here to help you with that.


Can you please share the issue you're facing with LPG or the gas supplier? Is it related to a new connection, existing supply, billing, or something else?


Human: i need gas only once


AI: It seems like you're looking for a one-time gas supply. You can approach the LPG distributor in your area and register for a one-time gas supply. However, please note that you may be required to pay a higher price for a one-time supply compared to a regular refill.


If you face any issues with the distributor or the supply, you can contact the Customer Service Cell of the company for assistance.


Additionally, you can also contact the National Consumer Helpline at 1800-11-4000 for immediate assistance with your consumer grievance.


Is there anything else I can help you with?


Human: the gas cylinder is faulty


AI: A faulty gas cylinder can be a safety hazard and a cause for concern. You should contact your LPG distributor or the company's Customer Service Cell to report the issue and request a replacement.


If the distributor is not responsive, you can also contact the LPG Emergency Helpline at 1906 (available 24*7) for assistance.


Additionally, you can also consider filing a complaint with the National Consumer Helpline at 1800-11-4000.


To facilitate the replacement process, it would be helpful to know the following:


1. What led to the issue with the gas cylinder (e.g., leak, rust, etc.)?
2. Do you have the receipt or the gas cylinder's serial number handy?
3. Have you already contacted the distributor or the Customer Service Cell regarding this issue?


Please let me know if there's anything else I can help you with.


Human: i'll call 1906. i don't know what led to the problem. i have serial number. i have not yet contacted service cell


AI: Calling the LPG Emergency Helpline at 1906 is a good first step. Since you're not sure what led to the issue, it's best to report it to the helpline and let them guide you further.


Having the serial number of the gas cylinder will be helpful in their investigation. Since you haven't contacted the service cell yet, the helpline can also assist you in escalating the issue to the next level if needed.


After speaking with the helpline, if you're not satisfied with the response or resolution, you can consider filing a complaint with the District Consumer Commission. You can find more information on the process on the National Consumer Dispute Redressal Commission (NCPCR) website: http://www.ncdrc.nic.in/districtlist.html.


If you need any further assistance or have any other questions, feel free to ask.


Also, don't forget to note down the reference number or case number provided by the helpline, as you may need it for future reference.


Is there anything else I can help you with?


Human: yes. no response on the helpline


AI: It can be frustrating when you don't get a response on the helpline. Let's try another option.


You can also contact the Customer Care number of the LPG distributor in your area. Here are some numbers for reference:


* Indian Oil: 1800-2333-555
* Bharat Petroleum: 1800-22-4344
* Hindustan Petroleum: [varies by region, you can check their website for the nearest customer care number]


If you're still not getting a response, you can also try contacting the company's main customer service number or the toll-free number for your state.


Additionally, you can also file a complaint on the MoPNG e-Seva portal or social media handles (Facebook: @MoPNGeSeva, Twitter: @MoPNG_eSeva) for a response.


If none of these options work, you can consider filing a complaint with the District Consumer Commission or the State Consumer Dispute Redressal Commission.


Do you want me to guide you through the process of filing a complaint with the District Consumer Commission?


Human: what is the toll free number for karnataka?


AI: For Karnataka, the toll-free numbers for the LPG distributors are:


* Indian Oil: 1800-2333-555
* Bharat Petroleum: 1800-22-4344
* Hindustan Petroleum: 1800-22-4000


You can try calling these numbers to report your issue and seek assistance.


If you're still facing issues, you can also contact the Karnataka State Consumer Dispute Redressal Commission (KSDRC) for guidance on filing a complaint.


The KSDRC can be contacted at:


Phone: 080-25580351, 25580352
Email: [karnatakastateconsumerdisputesredressalcommission@gmail.com](mailto:karnatakastateconsumerdisputesredressalcommission@gmail.com)


You can also visit their office in person or file a complaint online through their website.


Do you want me to guide you through the process of filing a complaint with the KSDRC?


Human: ok. where is ksdrc?


AI: The Karnataka State Consumer Dispute Redressal Commission (KSDRC) is located in Bangalore. The address is:


Karnataka State Consumer Dispute Redressal Commission,
3rd Floor, Vikasa Soudha,
Kendriya Sadan,
Bangalore - 560001.


You can also contact them at the phone numbers I provided earlier: 080-25580351, 25580352.


Additionally, you can visit their website at [www.karnatakastateconsumersdisputesredressalcommission.gov.in](http://www.karnatakastateconsumersdisputesredressalcommission.gov.in) for more information on the commission and the process of filing a complaint.


If you'd like to file a complaint in person, you can visit their office during working hours. It's always a good idea to call ahead and confirm their office hours and availability.


Do you have any other questions or would you like me to help you with anything else?


Human: that's too far. i live in bidar


AI: The KSDRC office in Bangalore might be a bit far from Bidar.


In that case, you can consider filing a complaint with the District Consumer Dispute Redressal Forum (DCDRF) in Bidar. The DCDRF is a subordinate to the KSDRC and has the power to hear and decide consumer disputes in the district.


You can contact the DCDRF in Bidar at:


District Consumer Dispute Redressal Forum,
Bidar,
Bidar District Court Complex,
Bidar - 585401.


Phone: 08482-252522


You can also visit their office in person or file a complaint online through their website.


Additionally, you can also contact the National Consumer Helpline at 1800-11-4000 for assistance and guidance on the process of filing a complaint.


Is there anything else I can help you with?


Human: tell me how to file complaint


AI: To file a complaint with the District Consumer Dispute Redressal Forum (DCDRF) in Bidar, you can follow these steps:


1. **Obtain a complaint form**: You can download the complaint form from the DCDRF's website or obtain it in person from their office. You can also ask for a printed form to be sent to you by post.
2. **Fill out the form**: Fill out the form with your details, including your name, address, contact number, and a brief description of the complaint.
3. **Attach supporting documents**: Attach any supporting documents that are relevant to your complaint, such as receipts, invoices, and photographs.
4. **Submit the form**: Submit the completed form to the DCDRF office in person or by post. If you submit it in person, make sure to get a receipt.
5. **Pay the fee**: Pay the required fee for filing the complaint, which is currently Rs. 100 for a consumer complaint.
6. **Wait for the acknowledgement**: The DCDRF will acknowledge your complaint and assign a case number.
7. **Wait for the hearing**: The DCDRF will schedule a hearing date and time for your complaint. You will be notified in writing.
8. **Attend the hearing**: Attend the hearing on the scheduled date and time and present your case.


Alternatively, you can also file a complaint online through the e-daakhil portal: http://edaakhil.nic.in/. This portal allows you to file a complaint online and track its status.


To file a complaint online, you will need to:


1. Register on the e-daakhil portal
2. Fill out the complaint form
3. Attach supporting documents
4. Pay the fee online
5. Submit the complaint


Please note that the e-daakhil portal is only available for filing complaints against entities that are registered on the portal. If the entity you are complaining against is not registered, you will need to file a complaint in person or by post.


Is there anything else I can help you with?


Human: give me the complaint form


AI: I'm a text-based AI and do not have the capability to directly provide you with a downloadable complaint form. However, I can guide you through the process of filling out a complaint form.


The District Consumer Dispute Redressal Forum (DCDRF) complaint form typically requires the following information:


**Complainant's Details**


* Name
* Address
* Contact Number
* Email ID (if available)


**Opposite Party's Details**


* Name
* Address
* Contact Number
* Email ID (if available)


**Complaint Details**


* Brief description of the complaint
* Date of the incident
* Amount claimed


**Relief Sought**


* What relief do you seek from the opposite party?


**Supporting Documents**


* Attach any relevant documents, such as receipts, invoices, photographs, etc.


Here's a sample complaint form that you can use as a reference:


**District Consumer Dispute Redressal Forum**
**Bidar**


**Complaint Form**


**Complainant's Details**


* Name: _____________________________________________________
* Address: __________________________________________________
* Contact Number: _____________________________________________
* Email ID: _______________________________________________


**Opposite Party's Details**


* Name: _____________________________________________________
* Address: __________________________________________________
* Contact Number: _____________________________________________
* Email ID: _______________________________________________


**Complaint Details**


* Brief description of the complaint: _____________________________________________________
* Date of the incident: ______________________________________________________
* Amount claimed: _______________________________________________________


**Relief Sought**


* What relief do you seek from the opposite party? _____________________________________________________


**Supporting Documents**


* Attach any relevant documents, such as receipts, invoices, photographs, etc.


You can use this as a sample and fill out the form with your details. Please note that this is not an official form and should be used as a reference only.


If you need a physical copy of the form, you can visit the DCDRF office in person or download it from their website.


Is there anything else I can help you with?


Human: what is dcdrf?


AI: DCDRF stands for District Consumer Dispute Redressal Forum. It is a quasi-judicial body that deals with consumer disputes at the district level. The DCDRF is responsible for hearing and deciding consumer complaints against businesses and service providers in a particular district.


The DCDRF is a part of the Consumer Protection Act, 1986, and is established by the state government. The DCDRF has the power to:


1. Hear and decide consumer complaints
2. Grant compensation to consumers
3. Order the opposite party to rectify the defect or deficiency in service
4. Impose penalties on the opposite party for non-compliance


The DCDRF has jurisdiction over consumer disputes that involve:


* Goods worth up to Rs. 1 crore
* Services worth up to Rs. 10 lakhs
* Goods and services provided within the district


The DCDRF has three members:


1. President
2. Vice-President
3. Member


The DCDRF follows a simple and speedy procedure to resolve consumer disputes. The hearing is usually conducted in a informal manner, and the decision is typically delivered within a few weeks.


In your case, since you are filing a complaint against an LPG distributor, the DCDRF in Bidar would be the appropriate forum to approach.


Is there anything else I can help you with?


Human: ok give me link to dcdrf site for complaint form


AI: Unfortunately, I'm a text-based AI and do not have the capability to provide direct links. However, I can guide you on how to find the DCDRF's website and download the complaint form.


To find the DCDRF's website, you can follow these steps:


1. Open a web browser (e.g., Google Chrome, Mozilla Firefox, etc.)
2. Type in the search bar: "District Consumer Dispute Redressal Forum Bidar"
3. Look for the official website of the DCDRF in Bidar, which should be a government website (e.g., [www.bidardcdrf.gov.in](http://www.bidardcdrf.gov.in) or similar)
4. On the website, look for a section that says "Forms" or "Downloads" and click on it
5. Search for the "Complaint Form" or "Application Form" and download it


Alternatively, you can also try searching for the DCDRF's website on the official website of the Government of Karnataka's Consumer Affairs Department:


* Website: [www.consumer.karnataka.gov.in](http://www.consumer.karnataka.gov.in)
* Look for the "District Consumer Dispute Redressal Forum" section and click on it
* Follow the same steps as above to download the complaint form


If you're still having trouble finding the website or the complaint form, you can contact the DCDRF office in Bidar directly for assistance.


Is there anything else I can help you with?''', 
'''AI: Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you.


Human: I took an Air India flight and my baggage was stuck at the airport for over an hour while we waited


AI: I'm so sorry to hear that you had a frustrating experience with Air India. Can you please tell me more about what happened? What led to your baggage being stuck at the airport for over an hour? Was it due to any specific reason or was it just a delay in the baggage handling process?


Human: They said that it was because the flight was parked very far from the airport


AI: Being far from the airport can indeed cause delays in baggage handling. However, the airline should still ensure that your baggage is handled efficiently and reaches you in a timely manner.


Can you please tell me what kind of relief are you seeking from Air India regarding this incident? Are you looking for a refund, compensation, or something else?


Human: They did not provide us with any refreshments or alternate methods of getting our bags. I missed a conference due to this. I want compensation for the trouble


AI: I understand that not only was your baggage delayed, but you also missed a conference due to the airline's inefficiency. As per the Air India's policy, if a flight is delayed by more than six hours, the airline should offer meals and refreshments to passengers. Since your flight was delayed, you should have been provided with refreshments.


Regarding the compensation, you are eligible for it since you missed a conference due to the airline's delay. The compensation amount can range between â‚¹5,000 to â‚¹10,000, depending on the block time (the time between the departure of a flight from the gate and arrival at the destination gate).


I can help you draft a letter to Air India for compensation. Would you like me to provide you with a draft letter? 


Also, I would like to inform you that you can also contact the National Consumer Helpline at 1800-11-4000 for immediate assistance. They can guide you through the process of filing a complaint and seeking compensation. Would you like me to provide you with the contact number of the local consumer helpline as well?


Human: Could you help me draft a letter to Air India?


AI: Here's a draft letter you can use as a template:


[Your Name]
[Your Address]
[City, State, PIN Code]
[Email ID]
[Phone Number]
[Date]


Air India
[Address]
[City, State, PIN Code]


Subject: Compensation Claim for Delayed Baggage and Missed Conference


Dear Sir/Madam,


Re: Flight [Flight Number] from [Departure City] to [Destination City] on [Date]


I am writing to bring to your notice a disturbing incident that occurred during my travel with Air India on [Date]. My baggage was stuck at the airport for over an hour due to the flight being parked far from the airport. Although I understand that this might have been a technical issue, I was not provided with any refreshments or alternate methods of getting my bags.


As a result, I missed a conference that I had scheduled for [Time] on [Date]. This has caused me significant inconvenience and financial loss.


As per the Air India's policy, I am entitled to compensation for the delay and inconvenience caused. I request you to provide me with compensation of â‚¹ [amount] for the trouble caused.


I would appreciate it if you could look into this matter urgently and take necessary steps to resolve it. I would also appreciate a response regarding the steps you will take to prevent such incidents in the future.


My PNR number is [PNR Number] and my booking reference number is [Booking Reference Number]. I have attached a copy of my ticket and a receipt for the conference I missed.


Thank you for your attention to this matter. I look forward to hearing from you soon.


Yours sincerely,


[Your Name]


Please note that you should fill in the brackets with your actual details and attach the required documents. Also, you can modify the letter as per your specific situation.


Also, don't forget to contact the National Consumer Helpline at 1800-11-4000 for immediate assistance. They can guide you through the process of filing a complaint and seeking compensation.


Is there anything else I can help you with?


Human: I am from Bangalore, but my flight was delayed in Bombay. Where do I submit my complaint?


AI: Since your flight was delayed in Bombay, you can submit your complaint to Air India's Mumbai office. However, I would recommend filing a complaint with the National Consumer Helpline (NCH) at 1800-11-4000 for immediate assistance. They can guide you through the process of filing a complaint and seeking compensation.


Additionally, you can also file a complaint on the Centralized Public Grievance Redress and Monitoring System (CPGRAMS) portal at [cpGRAMS.gov.in](http://cpGRAMS.gov.in). This portal is managed by the Ministry of Personnel, Public Grievances and Pensions, Department of Administrative Reforms and Public Grievances.


If you prefer to submit a physical complaint, you can visit the Air India office in Mumbai. Here is the address:


Air India Limited
Mumbai Office
A Wing, 12th Floor, 16/18, J.N. Heredia Marg, Ballard Estate,
Mumbai - 400 001.


Please note that you may need to make an appointment or contact them in advance to file a complaint.


Also, you can download the RailMadad App or visit the RailMadad portal at [railmadad.indianrailways.gov.in](http://railmadad.indianrailways.gov.in) to file a complaint.


Human: cant i file it with the bangalore office? it would be easier for me


AI: You can indeed file a complaint with the Air India office in Bangalore. It would be more convenient for you to visit their office in person and file a complaint.


Here is the address of the Air India office in Bangalore:


Air India Limited
Bangalore Office
No. 32, 4th Floor, Brigade Gateway, 26/27, Opp. Metro Station, Off. MG Road,
Bangalore - 560 025.


You can visit this office and file a complaint with their customer service team. They will guide you through the process and help you with your issue.


Before you visit, it's a good idea to call them and make an appointment or confirm their office hours to avoid any inconvenience.


Additionally, don't forget to keep a copy of your complaint and any supporting documents, such as your ticket and receipt for the conference you missed.


Is there anything else I can help you with?


Human: nothing else, thank you!


AI: It was my pleasure to assist you. I hope you get a resolution to your issue with Air India and receive the compensation you deserve. Don't hesitate to reach out if you need any further assistance.


Remember to also contact the National Consumer Helpline at 1800-11-4000 for any future consumer-related issues. They are available 24/7 to help you.


Have a great day and a smooth resolution to your issue!''',
'''AI: Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you.


Human: i got food poisoning


AI: I'm so sorry to hear that you've experienced food poisoning. That can be a very uncomfortable and concerning situation.


To help you, I'd like to know more about what happened. Can you please tell me what led to your food poisoning? Was it food from a restaurant, a shop, or something you cooked at home?


Human: it was a sandwich


AI: Food poisoning from a sandwich can be a real concern. To help you with your consumer grievance, I'd like to know more about the incident. Can you please tell me which shop or restaurant you bought the sandwich from?


Human: it ws from concu


AI: Concu is a popular food chain. I'd like to help you with your consumer grievance.


If you haven't already, I would recommend contacting the customer service team of Concu directly to report the incident and seek a resolution. If they are unresponsive or not helpful, you can also consider filing a complaint with the local consumer forum or district consumer dispute redressal forum in your area.


Before we proceed with any further steps, have you already tried contacting Concu's customer service team about this incident?


Human: they apologised but failed to compensate me


AI: It's disappointing when a company apologizes but doesn't take adequate action to make things right.


In this case, since Concu has already apologized but failed to compensate you, you may want to consider taking further action to seek a resolution. One option is to send a legal notice to Concu, detailing the incident, the relief you're seeking, and a deadline for them to respond and compensate you.


If you're not comfortable sending a legal notice yourself, I can help you draft a sample notice. Alternatively, you can also consider filing a complaint with the local consumer forum or district consumer dispute redressal forum in your area.


Before we proceed, would you like me to guide you on how to draft a sample legal notice or would you like to file a complaint with the consumer forum?


Human: i would like both


AI: I'll provide you with a sample legal notice and guide you on how to file a complaint with the consumer forum.


**Sample Legal Notice:**


[Your Name]
[Your Address]
[City, State, Pincode]
[Email ID]
[Mobile Number]
[Date]


Concu Food Chain
[Their Address]
[City, State, Pincode]


Subject: Notice for Compensation for Food Poisoning Incident


Dear Sir/Madam,


Re: Compensation for Food Poisoning Incident


This notice is to bring to your attention the incident of food poisoning that I, [Your Name], experienced after consuming a sandwich from your outlet located at [Outlet Address] on [Date]. Despite apologizing for the incident, you have failed to provide any compensation or adequate redressal.


The details of the incident are as follows:


* Date of consumption: [Date]
* Outlet location: [Outlet Address]
* Type of food consumed: Sandwich
* Symptoms experienced: [List symptoms]


I am seeking compensation of [Amount] for the suffering and expenses incurred due to the food poisoning incident. I request you to take immediate action to provide compensation and take necessary steps to prevent such incidents in the future.


You are required to respond to this notice within 7 days from the date of receipt of this notice, indicating your intention to provide compensation or take any other necessary action.


If I do not receive a satisfactory response within the stipulated time, I shall be forced to approach the consumer forum and take further action as per the Consumer Protection Act, 2019.


Yours faithfully,


[Your Name]


**Filing a Complaint with the Consumer Forum:**


To file a complaint with the consumer forum, you can follow these steps:


1. Visit the National Consumer Helpline website ([www.nchp.nic.in](http://www.nchp.nic.in)) or the e-daakhil portal ([edaakhil.nic.in](http://edaakhil.nic.in)).
2. Click on "File a Complaint" and select "Consumer Complaint".
3. Fill in the required details, including your name, address, mobile number, and email ID.
4. Provide details of the incident, including the date, time, and location of the incident, and the type of food consumed.
5. Specify the relief you are seeking, including compensation for medical expenses, suffering, and any other damages.
6. Attach supporting documents, such as medical bills, receipts, and any other relevant evidence.
7. Submit the complaint and wait for a response from the consumer forum.


Please note that you may need to visit the consumer forum in person to file the complaint, depending on the jurisdiction and the type of complaint.


If you have any questions or need further assistance, please feel free to ask.


Also, don't forget to contact the National Consumer Helpline at 1800-11-4000 for immediate assistance.


Is there anything else I can help you with?
''']
# chats.append("")
# chats.append("")
for j in range(1):
    print("Loop ", j)
    for i, chat in enumerate(chats[:], start=1):
        # In 5th (start 0) chat, there is Inconsistencies with the karnataka drc number and email id.
        # Wrong address for Air India Mumbai, and Air India Bangalore; addresses were completely fictional The bot is picking up compensation of Rs 5000-10,000 which is for not informing the flier 24 hours before flight, but the issue at hand is delay in baggage claim; the chatbot is also linking CPGRAMS portal, which is a correct grievance redressal mechanism, however the website is linked wrong, and has not been linked on our sectoral corpus; the chatbot is also linking the rail portal for some reason
        # No hallucination
        chat_sequence = process_chat_sequence(chat)
        res = response_analysis_chain.invoke({"chat":chat_sequence})
        print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan\n Chat ", i)
        print("Analysis-")
        print(res['analysis'])
        print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
        print("Result-")
        print(res['result'])
        print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
        print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
print("\nENDOFINFERENCE\n")

end_time = time.time()
print("End Time: ", end_time)
duration = end_time - start_time
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)
print(f"Time Taken: {hours:02}:{minutes:02}:{seconds:02}")