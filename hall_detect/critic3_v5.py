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
start_time = datetime.now()
print("Start Time: ", start_time)

def disp_dict(arr):
    print("Start Dictionary")
    print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x")
    for k,v in arr.items():
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
    def cutoff_at_stop_token(response, stop_token="<|endoftext|>"):
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

        # Prompt for the analysis
        analysis_prompt_template = PromptTemplate(
            template=(
                """You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot that provides correct and flawless legal advice to the user. You will analyze a chatbot's response based on the given information and identify any inconsistencies.

You are provided with:
- Context: The textual knowledge related to the query.
- History: The previous conversation between the user and the chatbot.
- Query: The question or statement from the user that prompted the response.
- Response: The chatbot’s reply to the query.

Evaluation Criteria:
You must determine whether the chatbot’s response contains inconsistencies based on the following:
1. Contradiction: Any part of the response that contradicts the given context and history.
2. Unsupported Information: Any facts or details that do not appear in the context or history but are presented as factual in the response.
3. Fabricated Details: Any made-up information such as contact numbers, addresses, legal provisions, or company names that are not in the context but present in the response.

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
- Missing assumptions in the context or conversation history are not inconsistencies.

4. Irrelevant Details
- Extra details that do not contradict the context should not be considered inconsistencies.
- Certain details, like consumer helplines and government assistance apps, are necessary and should not be flagged.

5. Missing Information
- If the chatbot omits a detail from the context but does not contradict it, this is not an inconsistency.

6. Partial Information
- If an inconsistency is flagged because the response provides only part of the context rather than the full information, it should not be considered an inconsistency as long as the given partial information is accurate.

7. Wrong Behavior
- If an inconsistency is about how the chatbot should behave or what it should have responded with, it is not an inconsistency as long as there is no contradiction with the context.
- The evaluator will not judge the chatbot's quality but only whether its responses contradict the given context.

Output Format:
Based on your evaluation, generate the following structured output:
1. Inconsistency Present: Yes or No
2. Inconsistencies: [List of inconsistencies only if Yes in Inconsistency Present. If No, leave it blank. The list of inconsistency should not belong to the above mentioned categories]
<|end_of_text|>

Remember to end with a <|end_of_text|> token

Example 1
Input:
Context:
Document 0:
The Employment Rights Act 1996 provides employees with certain rights, such as the right to request flexible working, protection against unfair dismissal, and statutory redundancy pay.

Document 1:
The Act does not mention any provisions about maternity benefits or sick pay directly.

History:
User: What rights do employees have under the Employment Rights Act 1996?
Chatbot: Employees have the right to flexible working, protection against unfair dismissal, and statutory redundancy pay.
User: Does it include maternity benefits?
Chatbot: Yes, the Act includes maternity benefits for employees.

Query:
Can you provide more details about maternity benefits under this Act?

Response:
Maternity benefits under the Employment Rights Act 1996 include 26 weeks of maternity leave and statutory maternity pay for eligible employees.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. The response states that the Employment Rights Act 1996 includes maternity benefits, but the context explicitly states that it does not mention them.
2. The details about “26 weeks of maternity leave” and “statutory maternity pay” are not supported by the context.
<|endoftext|>


Example 2
Input:
Context:
Document 0:
Under the Consumer Rights Act 2015, consumers are entitled to a refund, repair, or replacement for goods that are faulty or not as described.

Document 1:
The Act does not include any references to warranty periods being extended for defective goods.

History:
User: Can I get an extended warranty if my product is faulty under the Consumer Rights Act 2015?
Chatbot: Yes, the Act allows for an extended warranty period for defective goods.
User: What kind of extended warranty can I expect?
Chatbot: The warranty duration depends on the product type and its condition.

Query:
How long is the extended warranty period?

Response:
The Consumer Rights Act 2015 provides a 12-month extended warranty for defective goods.

Output:
first = TrueYes
Inconsistencies:
1. The response falsely claims that the Act provides for an extended warranty period, which is explicitly not mentioned in the context.
2. The "12-month extended warranty" detail is fabricated and unsupported by the context.
<|endoftext|>


Example 3
Input:
Context:
Document 0:
Section 21 of the Housing Act 1988 allows landlords to evict tenants without providing a reason, as long as the proper notice period is given.

Document 1:
The notice period is currently two months.

History:
User: Can a landlord evict a tenant without giving a reason under Section 21?
Chatbot: Yes, under Section 21, landlords can evict tenants without giving a reason.
User: What is the notice period for this kind of eviction?
Chatbot: I believe it’s one month, but I’ll confirm.

Query:
Can you confirm if one month is the required notice period?

Response:
Yes, the notice period under Section 21 is one month.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. The response incorrectly states that the notice period under Section 21 is one month, while the context explicitly states it is two months.
<|endoftext|>


Example 4
Input:
Context:
Document 0:
The Data Protection Act 2018 governs the processing of personal data in the UK. It requires organizations to obtain explicit consent for data processing and outlines individuals' rights regarding access, correction, and deletion of their data.

Document 1:
The Act does not detail penalties for non-compliance.

History:
User: What are the key points of the Data Protection Act 2018?
Chatbot: It governs personal data processing, mandates explicit consent, and provides rights for access, correction, and deletion of data.

Query:
Are there penalties for organizations that fail to comply?

Response:
The Data Protection Act 1995 governs personal data processing and requires explicit consent for such actions. While it outlines individual rights, it does not explicitly detail penalties for non-compliance.

Output:
Inconsistencies Present: No
Inconsistencies:
<|endoftext|>


Example 5
Input:
Context:
Document 0:
The Equal Pay Act 1970 prohibits discrimination between men and women in terms of pay and employment conditions.

Document 1:
The Act does not specify exceptions or provide details on enforcement mechanisms.

History:
User: Can employers pay men and women differently for the same job?
Chatbot: No, the Equal Pay Act 1970 prohibits such discrimination.

Query:
Are there any exceptions under this Act?

Response:
The Equal Pay Act 1970 prohibits pay discrimination between men and women for the same job. You can also contact +91-9823141241, the contact number of Women's Freedom Organisation for more information.

Output:
Inconsistencies Present: Yes
Inconsistencies:
1. The response introduces a Women's Freedom Organisation and its phone number, neither of which is present in the context.
<|endoftext|>


Based on these examples do the same for the following,

Input:
"""



                "Context:\n {context}. Consumer can also contact the National Consumer Helpline at 1800-11-4000 or UMANG App for more assistance\n"
                "History:\n {history}\n\n"
                "Query:\n {query}\n\n"
                "Response:\n {response}\n\nOutput:"
            ),
            input_variables=["context", "history", "query", "response"]
        )

        analysis_chain = analysis_prompt_template | llm2

        analysis_prompt_template2 = PromptTemplate(
            template=(
                """You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot. You have already identified whether inconsistencies exist in the chatbot’s response. Now, your goal is to analyze the identified inconsistencies, remove the wrong inconsistencies and assign a Degree of Inconsistency for each of the corrected inconsistencies. 

Input Provided to You:
1. Inconsistency Present: Yes or No
2. Inconsistencies: [List of identified inconsistencies]

You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot. You have already identified whether inconsistencies exist in the chatbot’s response. Now, your goal is to analyze the identified inconsistencies, remove the wrong inconsistencies, and assign a Degree of Inconsistency for each of the corrected inconsistencies.

Step 1: Removing Incorrect Inconsistencies
An inconsistency must be removed if it falls under any of the following conditions:

1. General Knowledge
- Statements about widely known facts, such as a company being a popular food chain or having customer service, should not be considered inconsistencies.
- The focus is on legal and consumer laws, not general knowledge.
- However, factual details like contact numbers, emails, addresses, and websites of companies, organizations, or government bodies are essential for legal discussions and should remain as inconsistencies.

2. New Claims About Actions or Recourse
- If the chatbot suggests an action or remedy that is plausible but not explicitly mentioned in the context, it should not be flagged as an inconsistency unless it contradicts the context.

3. Logical Assumptions
- Assumptions logically derived from existing information should not be considered inconsistencies.
- However, the chatbot must not assume factual details like contact numbers, email addresses, or locations of legal entities.
- Missing assumptions in the context or conversation history are not inconsistencies.

4. Irrelevant Details
- Extra details that do not contradict the context should not be considered inconsistencies.
- Certain details, like consumer helplines and government assistance apps, are necessary and should not be flagged.

5. Missing Information
- If the chatbot omits a detail from the context but does not contradict it, this is not an inconsistency.

6. Partial Information
- If an inconsistency is flagged because the response provides only part of the context rather than the full information, it should not be considered an inconsistency as long as the given partial information is accurate.

7. Wrong Behavior
- If an inconsistency is about how the chatbot should behave or what it should have responded with, it is not an inconsistency as long as there is no contradiction with the context.
- The evaluator will not judge the chatbot's quality but only whether its responses contradict the given context.

After applying the above criteria, remove inconsistencies that are invalid and retain the remaining ones.

Assigning Degree of Inconsistency
Each valid inconsistency must be assigned a degree from 1 to 5, based on its severity.

Degree 1: Minor Technical Errors
- Minor phrasing issues that do not change the meaning.
- Slight variations in wording that do not impact legal accuracy.

Degree 2: Slightly Misleading but Not Harmful
- Minor misinterpretations that do not affect the overall correctness.
- Incorrect terminology that does not misguide the user significantly.

Degree 3: Noticeable Errors but Limited Impact
- Errors in minor procedural details that do not prevent the user from taking correct action.
- Providing an incomplete legal explanation while still giving a correct overall direction.

Degree 4: Serious Misleading Information
- Partially incorrect legal or procedural information that could lead to misunderstandings.
- Minor errors in contact details such as a slightly incorrect phone number, website, or address.
- Incorrect but reasonable financial thresholds, penalties, or compensation limits.
- Contextual misinterpretation leading to a suggestion that is not the best course of action.
- Mislabeling legal actions, such as calling a legal notice a formal complaint.

Degree 5: Critical Errors (Severe Hallucinations)
- Completely fabricated legal processes, rules, or authorities.
- False or fictional legal remedies or statutory provisions.
- Incorrect or fictional contact information, such as fake government office addresses, phone numbers, emails, or websites.
- Misdirecting users to the wrong legal body, leading them to file complaints incorrectly.
- Fundamental misrepresentation of laws or legal jurisdictions.
- Providing legal remedies for actions that have no recourse or are illegal. 



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

Ensure the output always ends with an `<|end_of_text|>` token


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
Explanation: The chatbot assumes a specific refund period without any supporting information, which may cause confusion.
<|end_of_text|>


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
Explanation: Assigning incorrect jurisdiction to a regulatory authority may cause users to pursue the wrong legal recourse.
<|end_of_text|>


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
Explanation: Giving wrong factual information like a fake contact number can severely mislead users and obstruct resolution.
<|end_of_text|>


Example 4
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response claims that the Food Safety and Standards Authority handles restaurant billing disputes, but this is not stated in the context.
2. The response provides a method to escalate complaints through the legal system, but the exact process is not given in the context.
3. The response states that users can email complaints to fssaicomplaints@india.org, but this email is not verified.
4. The response states that restaurant disputes can be reported through the Consumer Protection Board, but this is not mentioned in the context.

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
Explanation: While the Consumer Protection Board may be relevant, its role in restaurant disputes is not explicitly mentioned and could be misleading.
<|end_of_text|>


Example 5
Input:
Inconsistency Present: No
Inconsistencies:

Output:
Inconsistency Present: No
Explanation for Correction: No inconsistencies present
Inconsistencies:
Degree of Inconsistency: 1
Explanation: No inconsistencies present
<|end_of_text|>


Now based on these examples work for the following,
Input:
"""
                    


                "{inconsistency}\n\nOutput:"
            ),
            input_variables=["inconsistency"]
        )

        analysis_chain2 = analysis_prompt_template2 | llm2

        # Define a custom chain that combines reformulating and analysis
        class ResponseAnalysisChain:
            def __init__(self, retriever, reformulate_chain, analysis_chain):
                self.retriever = retriever
                self.reformulate_chain = reformulate_chain | StrOutputParser()
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

                # Retrieve relevant context
                context_docs = self.retriever.get_relevant_documents(reformulated_query)
                context = "".join([f"\nDocument {doc_num}:\n{doc.page_content.replace("\n", " ")}" for doc_num, doc in enumerate(context_docs)])

                # Pass everything to the analysis chain
                final_inputs = {
                    "context": context,
                    "history": history,
                    "query": query,
                    "reformulated": reformulated_query,
                    "response": response,
                }
                analysis_temp = cutoff_at_stop_token(self.analysis_chain.invoke(final_inputs))
                if "Inconsistencies Present: No" in analysis_temp:
                    analysis = analysis_temp
                else:
                    analysis = cutoff_at_stop_token(self.analysis_chain2.invoke({"inconsistency":analysis_temp}))
                final_inputs['analysis_temp'] = analysis_temp
                final_inputs['analysis'] = analysis
                # final_inputs.pop('history')
                # final_inputs.pop('reformulated')
                disp_dict(final_inputs)
                return analysis, final_inputs

        return ResponseAnalysisChain(retriever, reformulate_chain, analysis_chain)

    def detect_hallucination_per_chat(chain, llm):
        # Define a function that performs the analysis dynamically
        def transform_fn(inputs):
            chat = inputs["chat"]  # Expect the input to have a key "chat"
            analysis = []
            analysis_test = []
            chat_logs = []
            for i, chat_triple in enumerate(chat, start=1):
                result, final_inputs = chain(chat_triple)
                # Extract the degree of inconsistency from the result (assuming it's explicitly stated in the result text).
                # analysis_test.append(result)
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
                chat_logs.append(final_inputs)
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
            analysis_tex = '''Turn 2
Inconsistencies Present: Yes
Inconsistencies:
1. The response introduces the National Consumer Helpline at 1800-11-4000, which is not mentioned in the context. Degree of Inconsistency is 4

Turn 4
Inconsistencies Present: Yes
Inconsistencies:
2. The response provides a link to the NCDRC website, but the context mentions the District Consumer Commission, not NCDRC. Degree of Inconsistency is 4
3. The response mentions that the consumer can contact the LPG Emergency Helpline at 1906, but the context states that the emergency helpline is available 24*7 operations with 2 shifts 12 hours each for attending emergency LPG leakage complaints. Degree of Inconsistency is 5

Turn 6
Inconsistencies Present: Yes
Inconsistencies:
1. The response mentions the toll-free numbers for LPG distributors in Karnataka but does not include Hindustan Petroleum's number as mentioned in the context. Instead, it incorrectly lists Hindustan Petroleum's number as 1800-22-4000. Degree of Inconsistency is 5
3. The response mentions the toll-free numbers for LPG distributors in Karnataka but does not include Hindustan Petroleum's number as mentioned in the context. Instead, it incorrectly lists Hindustan Petroleum's number as 1800-22-4000. Degree of Inconsistency is 5

Turn 7
Inconsistencies Present: Yes
Inconsistencies:
1. The response provides the address of the KSDRC as "Vikasa Soudha, Kendriya Sadan, Bangalore - 560001", which is not accurate according to the context. Degree of Inconsistency is 5
2. The response mentions visiting the KSDRC website at "www.karnatakastateconsumersdisputesredressalcommission.gov.in", which is not correct as per the context. Degree of Inconsistency is 5
3. The response provides phone numbers for the KSDRC as "080-25580351, 25580352", which are not accurate according to the context. Degree of Inconsistency is 5
5. The response provides incorrect information about the KSDRC office hours and availability. Degree of Inconsistency is 4

Turn 8
Inconsistencies Present: Yes
Inconsistencies:
1. The response mentions that the DCDRF in Bidar has the power to hear and decide consumer disputes in the district, which is not accurate. The DCDRF has the power to hear and decide consumer disputes in the state, not just the district. Degree of Inconsistency is 4
2. The response provides a website for the DCDRF in Bidar, which is not a valid website. The correct website is not provided in the context. Degree of Inconsistency is 5

Turn 9
Inconsistencies Present: Yes
Inconsistencies:
1. The response mentions the e-daakhil portal but does not provide the correct link for filing a complaint. Degree of Inconsistency is 4
2. The response does not provide the correct address of the DCDRF office in Bidar. Degree of Inconsistency is 4
3. The response mentions the e-daakhil portal but does not provide the correct link for filing a complaint. Degree of Inconsistency is 4
4. The response does not provide the correct address of the DCDRF office in Bidar. Degree of Inconsistency is 4

Turn 10
Inconsistencies Present: Yes
Inconsistencies:
1. The response does not provide the actual complaint form as requested by the user. Degree of Inconsistency is 4
2. The response mentions that the user can download the complaint form from the DCDRF's website or obtain it in person from their office, but it does not provide the correct website or office address. Degree of Inconsistency is 5

Turn 11
Inconsistencies Present: Yes
Inconsistencies:
1. The response introduces the DCDRF as a quasi-judicial body that deals with consumer disputes at the district level, but the context only mentions the DCDRF in Bidar and does not provide information about its jurisdiction, powers, or composition. Degree of Inconsistency is 5
2. The response mentions that the DCDRF has jurisdiction over consumer disputes involving goods worth up to Rs. 1 crore and services worth up to Rs. 10 lakhs, but the context does not provide any information about the jurisdiction of the DCDRF in Bidar. Degree of Inconsistency is 4
3. The response states that the DCDRF has three members, but the context does not provide any information about the composition of the DCDRF in Bidar. Degree of Inconsistency is 4

Turn 12
Inconsistencies Present: Yes
Inconsistencies:
2. The response suggests contacting the DCDRF office in Bidar directly for assistance, which is not a necessary step as the website is publicly available. Degree of Inconsistency is 4
3. The response does not provide the correct website address for the DCDRF in Bidar, which is a crucial piece of information for filing a complaint. Degree of Inconsistency is 5
'''
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
3. [Third inconsistency]
<|end_of_text|>

- If no inconsistencies are detected:
Inconsistencies detected: No.
<|end_of_text|>

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
3. The response suggests that the refund process takes 30 days, which is not supported by the context.
<|end_of_text|>


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
2. The response claims that a consumer complaint can be filed via email, but no such method is mentioned in the context.
<|end_of_text|>


Example 3:
Input:
Analysis:
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
            return {"analysis": analysis_text,"result": cutoff_at_stop_token(combination_chain.invoke({"analysis": analysis_text})), "logs":chat_logs}
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

# Create the RAG-based QA chain
print("START")
ver = "1"
# Example query
data = pd.read_csv("./hall_detect/data1.csv", index_col=0)
data.dropna(inplace=True, subset = ["Chat"])
data.reset_index(inplace=True, drop=True)
chats = data["Chat"].to_list()

response_analysis_chain = create_rag_qa_chain_with_response_analysis(folder_path)
output_file = "./hall_detect/critic3_v5/file_output_v" +ver +".txt"
final_df_loc = "./hall_detect/critic3_v5/result_df_v" +ver +".csv"
csv_file = "./hall_detect/critic3_v5/output_df_v" +ver +".csv"
with open(output_file, "a") as f:
    f.write("Start\n\n\n\n\n\n")  
result = []
try:
    df_final = pd.read_csv(final_df_loc, index_col=0)
    print("Starting from ", len(df_final))
except:
    df_final = pd.DataFrame(columns=["Chat", "Result", "Logs"])
    print("Starting from 0")
df_final_list = df_final.to_dict("records")
for chat_no, chat in enumerate(chats[len(df_final):], start=len(df_final)):
    print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x")
    print("START CHAT " + str(chat_no))
    res = {'analysis': "Sample analysis text", 'result': "Sample result text", 'logs': "Sample logs text"}
    res = response_analysis_chain.invoke({"chat": process_chat_sequence(chat)})
    disp_dict({'Analysis':res['analysis'], 'Result':res['result']})
    result.append(res['result'])
    df_final_list.append({"Chat": chat, "Result": res['result'], "Logs": res['logs']})
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
