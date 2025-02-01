import os
from PyPDF2 import PdfReader
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.transform import TransformChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

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
        from langchain_huggingface import HuggingFacePipeline
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

        # Initialize the LLM for text generation
        callbacks = [StreamingStdOutCallbackHandler()]
        llm1 = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            task="text-generation",
            device=0,  # Use GPU if available
            callbacks=callbacks,  # For streaming outputs
            pipeline_kwargs=dict(
                return_full_text=False,  # Return only the new tokens
                max_new_tokens=512,  # Limit the number of generated tokens
                do_sample=True,  # Enable sampling for varied outputs
                temperature=0.5,  # Balance randomness and coherence
                repetition_penalty = 1.02,
            ),
            # model_kwargs=dict(stop=["<|endoftext|>"], ) # Add stop tokens here
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
2. Unsupported Information: Any facts or details that do not appear in the context or history but are presented as factual.
3. Fabricated Details: Any made-up information such as contact numbers, addresses, legal provisions, or company names that are not in the context.

These are not inconsistencies and should not be detected as such:
1. Irrelevant details: Details not requested by the user but not contradicting the context are not inconsistencies
2. Missing Information: Any details which are not mentioned even though present in context is not an inconsistency as long as it does not contradict the context or history
3. Natural Assumptions: Any assumptions which are natural and not contradicting the context or history which might include user wants and preferences.

Output Format:
Based on your evaluation, generate the following structured output:
1. Inconsistency Present: Yes or No
2. Inconsistencies: [List of inconsistencies only if Yes in Inconsistency Present. If No, leave it blank]

Example 1:
Context:
The Employment Rights Act 1996 provides employees with certain rights, such as the right to request flexible working, protection against unfair dismissal, and statutory redundancy pay. It does not mention any provisions about maternity benefits or sick pay directly.

History:
User: What rights do employees have under the Employment Rights Act 1996?
Chatbot: Employees have the right to flexible working, protection against unfair dismissal, and statutory redundancy pay.
User: Does it include maternity benefits?
Chatbot: Yes, the Act includes maternity benefits for employees.

Query:
Can you provide more details about maternity benefits under this Act?

Response:
Maternity benefits under the Employment Rights Act 1996 include 26 weeks of maternity leave and statutory maternity pay for eligible employees.

Inconsistency Present: Yes
Inconsistencies:
1. The response states that the Employment Rights Act 1996 includes maternity benefits, but the context explicitly states that it does not mention them.
2. The details about “26 weeks of maternity leave” and “statutory maternity pay” are not supported by the context.
<|endoftext|>
Example 2:
Context:
Under the Consumer Rights Act 2015, consumers are entitled to a refund, repair, or replacement for goods that are faulty or not as described. The Act does not include any references to warranty periods being extended for defective goods.

History:
User: Can I get an extended warranty if my product is faulty under the Consumer Rights Act 2015?
Chatbot: Yes, the Act allows for an extended warranty period for defective goods.
User: What kind of extended warranty can I expect?
Chatbot: The warranty duration depends on the product type and its condition.

Query:
How long is the extended warranty period?

Response:
The Consumer Rights Act 2015 provides a 12-month extended warranty for defective goods.

Inconsistencies Present: Yes
Inconsistencies:
1. The response falsely claims that the Act provides for an extended warranty period, which is explicitly not mentioned in the context.
2. The "12-month extended warranty" detail is fabricated and unsupported by the context.
<|endoftext|>

Example 3:
Context:
Section 21 of the Housing Act 1988 allows landlords to evict tenants without providing a reason, as long as the proper notice period is given. The notice period is currently two months.

History:
User: Can a landlord evict a tenant without giving a reason under Section 21?
Chatbot: Yes, under Section 21, landlords can evict tenants without giving a reason.
User: What is the notice period for this kind of eviction?
Chatbot: I believe it’s one month, but I’ll confirm.

Query:
Can you confirm if one month is the required notice period?

Response:
Yes, the notice period under Section 21 is one month.

Inconsistencies Present: Yes
Inconsistencies:
1. The response incorrectly states that the notice period under Section 21 is one month, while the context explicitly states it is two months.
<|endoftext|>

Example 4:
Context:
The Data Protection Act 2018 governs how personal data should be processed and stored. It includes principles such as lawfulness, fairness, transparency, and data minimization. The Act does not mention specific fines for breaches.

History:
User: What is the main purpose of the Data Protection Act 2018?
Chatbot: The Act ensures personal data is processed lawfully and transparently.
User: Are there any fines for breaches mentioned in the Act?
Chatbot: Yes, there are specific fines.

Query:
What are the fines for breaches under this Act?

Response:
The Data Protection Act 2018 specifies fines of up to £20 million for breaches.

Inconsistencies Present: Yes
Inconsistencies:
1. The response falsely claims that the Data Protection Act 2018 specifies fines of up to £20 million, while the context explicitly states that the Act does not mention specific fines.
<|endoftext|>

Example 5:
Context:
The Road Traffic Act 1988 mandates that all drivers must have third-party liability insurance. It does not specify penalties for not having insurance.

History:
User: Does the Road Traffic Act 1988 require insurance?
Chatbot: Yes, all drivers must have third-party liability insurance.
User: What happens if someone doesn’t have insurance?
Chatbot: I’m not sure about the exact penalties.

Query:
Can you confirm the penalties under this Act?

Response:
Under the Road Traffic Act 1988, driving without insurance results in a penalty of £300 and six points on the driver’s license.

Inconsistencies Present: Yes
Inconsistencies:
1. The response incorrectly states that the penalty is mentioned in the Act, whereas the context explicitly states that the Act does not specify penalties.
<|endoftext|>

Example 6:
Context:
The Health and Safety at Work Act 1974 ensures the safety, health, and welfare of employees at work. Employers must conduct regular risk assessments and provide necessary safety equipment. The Act does not mention mental health initiatives or stress management requirements explicitly.

History:
AI: Hello! How can I assist with workplace safety issues?
User: My employer doesn't provide safety equipment.
AI: Employers are required to provide necessary safety equipment under the Health and Safety at Work Act 1974. Can you share more about the equipment missing or the risks involved?

Query:
The equipment is missing entirely.

Response:
If your employer has not provided any safety equipment, they are in violation of the Health and Safety at Work Act 1974. You can report this to the Health and Safety Executive (HSE). Additionally, the Act mandates that employers must address workplace stress and mental health issues, which can also be reported.

Inconsistencies Present: Yes
Inconsistencies:
1. The response claims that the Act mandates addressing workplace stress and mental health issues, which is not supported by the context.
<|endoftext|>

Example 7:
Context:
The Consumer Protection Act 1987 ensures that consumers are protected against defective products that cause injury or harm. The Act outlines the liability of manufacturers but does not detail specific compensation amounts or procedures.

History:
User: What does the Consumer Protection Act 1987 cover?
Chatbot: It ensures protection against defective products that cause injury or harm and specifies manufacturer liability.

Query:
Does it specify compensation amounts?

Response:
The Consumer Protection Act 1987 does not specify exact compensation amounts but establishes liability for manufacturers, allowing injured parties to seek compensation through the courts.

Inconsistencies Present: No
Inconsistencies:
<|endoftext|>

Example 8:
Context:
The Employment Rights Act 1996 outlines employees' rights, including protection from unfair dismissal and the right to statutory redundancy pay. It does not mention provisions related to remote work or flexible working hours.

History:
User: Does the Employment Rights Act cover flexible working hours?
Chatbot: No, the Act does not mention provisions for flexible working hours.

Query:
Can you clarify how the Act protects employees?

Response:
The Employment Rights Act 1996 protects employees by outlining their rights against unfair dismissal and providing statutory redundancy pay. It does not cover flexible working hours.

Inconsistencies Present: No
Inconsistencies:
<|endoftext|>

Example 9:
Context:
The Data Protection Act 2018 governs the processing of personal data in the UK. It requires organizations to obtain explicit consent for data processing and outlines individuals' rights regarding access, correction, and deletion of their data. It does not detail penalties for non-compliance.

History:
User: What are the key points of the Data Protection Act 2018?
Chatbot: It governs personal data processing, mandates explicit consent, and provides rights for access, correction, and deletion of data.

Query:
Are there penalties for organizations that fail to comply?

Response:
The Data Protection Act 2018 governs personal data processing and requires explicit consent for such actions. While it outlines individual rights, it does not explicitly detail penalties for non-compliance.

Inconsistencies Present: Yes
Inconsistencies:
<|endoftext|>

Example 10:
Context:
The Equal Pay Act 1970 prohibits discrimination between men and women in terms of pay and employment conditions. It does not specify exceptions or provide details on enforcement mechanisms.

History:
User: Can employers pay men and women differently for the same job?
Chatbot: No, the Equal Pay Act 1970 prohibits such discrimination.

Query:
Are there any exceptions under this Act?

Response:
The Equal Pay Act 1970 prohibits pay discrimination between men and women for the same job. However, exceptions might exist if justified by factors such as seniority or performance.

Inconsistencies Present: Yes
Inconsistencies:
1. The response introduces assumptions about possible exceptions (e.g., seniority or performance) that are not mentioned in the context.
<|endoftext|>
"""



                "Context:\n {context}\nIt is better to contact the National Consumer Helpline (1800-11-4000) or UMANG App for more assistance\n"
                "History:\n {history}\n\n"
                "Query:\n {query}\n\n"
                "Response:\n {response}\n\n"
            ),
            input_variables=["context", "history", "query", "response"]
        )

        analysis_chain = analysis_prompt_template | llm2

        analysis_prompt_template2 = PromptTemplate(
            template=(
                """You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot. You have already identified whether inconsistencies exist in the chatbot’s response. Now, your goal is to analyze the identified inconsistencies, remove the wrong inconsistencies and assign a Degree of Inconsistency to the response.

Input Provided to You:
1. Inconsistency Present: Yes or No
2. Inconsistencies: [List of identified inconsistencies]

Conditions for Wrong Inconsistency:
1. General Knowledge: The inconsistency is based upon general knowledge and not with regards to legal or consumer laws. The context only deals with legal knowledge and hence the inconsistencies detected are incorrect. General Knowledge does not include information based upon details of a company, organisation or government body as they are required to be present in the context to be deemed correct.
2. Logical Assumptions: Assumptions derived using logic and reason from whatever knowledge present are not inconsistencies. These assumptions are not wrong as conversations are full of assumptions which are all correct unless they clearly contradict the context. Assumptions missing in context or history are not inconsistencies.
3. Partial Information: If inconsistency is due to the full information in the context not being in the response but rather only a partial or incomplete information then it is not an inconsistency as long as the given partial information does not have inaccuracies.
Rewrite the list by removing the wrong inconsistency.

Degree of Inconsistency Rules:
1. Degree 1: No inconsistencies. The response aligns perfectly with the context.
2. Degree 2: Minor unsupported assumptions that are plausible and do not significantly impact accuracy.
3. Degree 3: Noticeable unsupported claims or minor contradictions with the context.
4. Degree 4: Clear contradictions or fabrications that impact accuracy but do not entirely undermine the response.
5. Degree 5: Severe contradictions or fabrications that completely misrepresent the context.

Output Format:
Based on the inconsistencies provided, generate the following structured output:
1. Degree of Inconsistency: [1 to 5]
2. Explanation: [Short and concise reason based on the listed inconsistencies]

Example 1:
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response states that the Employment Rights Act 1996 includes maternity benefits, but the context explicitly states that it does not mention them.
2. The response assumes that the user wants to avail the maternity benefits but this has not been specified in the history or context.
3. The details about “26 weeks of maternity leave” and “statutory maternity pay” are not supported by the context.


Output:
Inconsistency Present: Yes
Inconsistencies:
1. The response states that the Employment Rights Act 1996 includes maternity benefits, but the context explicitly states that it does not mention them.
2. The details about “26 weeks of maternity leave” and “statutory maternity pay” are not supported by the context.
Degree of Inconsistency: 5
Explanation: The response fabricates maternity leave details not supported by the context and falsely claims that the Act includes these benefits. This is a major inconsistency.
<|endoftext|>

Example 2:
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response falsely claims that the Act provides for an extended warranty period, which is explicitly not mentioned in the context.
2. The "12-month extended warranty" detail is fabricated and unsupported by the context.

Output:
Inconsistency Present: Yes
Inconsistencies:
1. The response falsely claims that the Act provides for an extended warranty period, which is explicitly not mentioned in the context.
2. The "12-month extended warranty" detail is fabricated and unsupported by the context.
Degree of Inconsistency: 5
Explanation: The response fabricates a "12-month extended warranty" detail not supported by the context and falsely claims the Act includes extended warranty provisions. This is a major inconsistency.
<|endoftext|>

Example 3:
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response incorrectly states that the notice period under Section 21 is one month, while the context explicitly states it is two months.
2. The response states that the user can ask more doubts but this has never been mentioned in the context or history.

Output:
Inconsistency Present: Yes
Inconsistencies:
1. The response incorrectly states that the notice period under Section 21 is one month, while the context explicitly states it is two months.
Degree of Inconsistency: 4
Explanation: The response provides an incorrect notice period of one month, directly contradicting the context that states it is two months. This significantly impacts the accuracy of the response.
<|endoftext|>

Example 4:
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response falsely claims that the Data Protection Act 2018 specifies fines of up to £20 million, while the context explicitly states that the Act does not mention specific fines.
2. The response mentions that the user wants to get back the damages he has encountered but this is not supported by context.

Output:
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response falsely claims that the Data Protection Act 2018 specifies fines of up to £20 million, while the context explicitly states that the Act does not mention specific fines.
Degree of Inconsistency: 5
Explanation: The response introduces fabricated details about fines of up to £20 million, which are not mentioned in the context. This is a major inconsistency.
<|endoftext|>

Example 5:
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response incorrectly states that the penalty is mentioned in the Act, whereas the context explicitly states that the Act does not specify penalties.

Output:
Inconsistency Present: Yes
Inconsistencies:
1. The response incorrectly states that the penalty is mentioned in the Act, whereas the context explicitly states that the Act does not specify penalties.
Degree of Inconsistency: 5
Explanation: The response fabricates details about penalties that are explicitly not mentioned in the context. This is a major inconsistency.
<|endoftext|>

Example 6:
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response claims that the Act mandates addressing workplace stress and mental health issues, which is not supported by the context.
2. The response claims that the user has been experiencing stress and gives solutions to it but this has not been mentioned in the context or history.

Output:
Inconsistency Present: Yes
Inconsistencies:
1. The response claims that the Act mandates addressing workplace stress and mental health issues, which is not supported by the context.
Degree of Inconsistency: 4
Explanation: The response incorrectly states that the Act mandates addressing workplace stress and mental health issues, introducing unsupported details. This is a significant inconsistency.
<|endoftext|>

Example 7:
Input:
Inconsistency Present: No
Inconsistencies:

Output:
Inconsistency Present: No
Inconsistencies:
Degree of Inconsistency: 1
Explanation: The response aligns perfectly with the context.
<|endoftext|>

Example 8:
Input:
Inconsistency Present: No
Inconsistencies:

Output:
Inconsistency Present: No
Inconsistencies:
Degree of Inconsistency: 1
Explanation: The response is consistent with the context.
<|endoftext|>

Example 9:
Input:
Inconsistency Present: Yes
Inconsistencies:
1. The response introduces assumptions about possible exceptions (e.g., seniority or performance) that are not mentioned in the context.

Output:
Inconsistency Present: Yes
Inconsistencies:
1. The response introduces assumptions about possible exceptions (e.g., seniority or performance) that are not mentioned in the context.
Degree of Inconsistency: 3
Explanation: The response introduces assumptions about exceptions that are not explicitly stated in the context. While plausible, these assumptions extend beyond the provided information and introduce noticeable inaccuracies.
<|endoftext|>


Input:
"""
                    


                "{inconsistency}"
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
                context = " ".join([doc.page_content for doc in context_docs])

                # Pass everything to the analysis chain
                final_inputs = {
                    "context": context.replace("\n", " "),
                    "history": history,
                    "query": query,
                    "reformulated": reformulated_query,
                    "response": response,
                }
                analysis_temp = cutoff_at_stop_token(self.analysis_chain.invoke(final_inputs))
                analysis = cutoff_at_stop_token(self.analysis_chain2.invoke({"inconsistency":analysis_temp}))
                final_inputs['analysis'] = analysis_temp + '\n-----------------\n' + analysis
                final_inputs.pop('history')
                final_inputs.pop('reformulated')
                disp_dict(final_inputs)
                global first
                if first==True:
                    # print(final_inputs['context'].replace('/n', ' '))
                    first = False
                return analysis

        return ResponseAnalysisChain(retriever, reformulate_chain, analysis_chain)

    def detect_hallucination_per_chat(chain, llm):
        # Define a function that performs the analysis dynamically
        def transform_fn(inputs):
            chat = inputs["chat"]  # Expect the input to have a key "chat"
            analysis = []
            for i, chat_triple in enumerate(chat[1:]):
                result = chain(chat_triple)
                # Extract the degree of inconsistency from the result (assuming it's explicitly stated in the result text).
                degree_line = [line for line in result.split("\n") if "Degree of Inconsistency:" in line]
                if degree_line:
                    degree = int(degree_line[0].split(":")[1].strip())
                    if degree >= 0:  # Include only if degree is 4 or 5
                        analysis.append("\nTurn " + str(i + 1) + "\n" + result)
            analysis_text = "".join(analysis)
            
            # Create the analysis prompt
            prompt = PromptTemplate(
                template=(
                    # "You are given an individual history of Inconsistencies detected in a chatbot. You are an evaluator and your task is to use all of these statements and give out simple description answering whether Inconsistencies are present as a whole in the chatbot. Do not individually mention each Inconsistencies but give a single statement describing whether Inconsistencies are present or not. It should be in this format:"
                    # "Inconsistencies are present"
                    # "The folllowing information is inconsistent"
                    # "\nAnswer in short. What is the final response to the questions: Are there any Inconsistencies in this chat? List the most significant of them"
                    "You are provided with a record of inconsistencies detected during interactions with a chatbot across multiple turns. As an evaluator, your task is to analyze all the detected inconsistencies collectively and provide a concise assessment of whether inconsistencies are present in the chatbot’s responses as a whole.\n"
                    "You are given multiple turns and you are to give a response for those turns only and not generate additional turns. Your task includes making an individual assessment of each inconistency and evaluating whether they are serious inconsistencies or not\n"
                    "Serious inconsistencies are inconsistencies that Contain information that is incorrect or opposes the context. Inconsistencies are not serious if they are due to assumptions that are safe to assume for instance that the company is registered or that the user wants a remedy.\n"
                    "Your response should not list individual serious inconsistencies but instead deliver a unified summary based on the entire chat. Use one of the following formats to present your evaluation:\n"
                    "- 'Inconsistencies detected: Yes.'\n"
                    "  'The following information is inconsistent: [briefly describe the most significant inconsistencies].'\n"
                    "- 'Inconsistencies detected: No.'\n"
                    "Ensure that the response is concise, without repetition and ends with an `<|endoftext|>` token"
                    "Provide a short and clear final answer to these questions using tha:\n"
                    "- Are there any inconsistencies in this chat?\n"
                    "- What are the most significant inconsistencies observed?\n"
                    "Here are a few examples"
                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The response incorrectly states that the Consumer Protection Act 2019 came into effect on August 9, 2019, whereas it was only notified on that date and came into effect on July 20, 2020.\n"
                    "2. The response claims that the Consumer Protection Act 1986 is still valid, which contradicts the fact that it was repealed by the 2019 Act.\n"
                    "Turn 2\n"
                    "Inconsistencies Present: No\n"
                    "Inconsistencies:\n"
                    "Turn 3\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The response states that the Consumer Protection Act 2019 does not apply to e-commerce platforms, which is incorrect as the Act specifically includes e-commerce and direct selling transactions.\n"
                    "Turn 4\n"
                    "Inconsistencies Present: No\n\n"

                    "Output:\n"
                    "Inconsistencies detected: Yes\n"
                    "1. The chatbot incorrectly states that the Consumer Protection Act 2019 came into effect on August 9, 2019, when it was only notified on that date and came into effect on July 20, 2020.\n"
                    "2. The chatbot claims the Consumer Protection Act 1986 is still valid, contradicting the fact that it was repealed by the 2019 Act.\n"
                    "3. The chatbot states that the Consumer Protection Act 2019 does not apply to e-commerce platforms, which is incorrect as the Act includes e-commerce and direct selling transactions.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: No\n"
                    "Turn 2\n"
                    "Inconsistencies Present: No\n"
                    "Turn 3\n"
                    "Inconsistencies Present: No\n\n"

                    "Output:\n"
                    "Inconsistencies detected: No\n"
                    "No inconsistencies are present.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The response incorrectly states that the Right to Consumer Awareness is a new right introduced by the Consumer Protection Act 2019. However, this right was already recognized under the Consumer Protection Act 1986.\n"
                    "Turn 2\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The chatbot claims that consumers must physically visit consumer courts to file complaints, but the 2019 Act allows for e-filing of complaints.\n"
                    "Turn 3\n"
                    "Inconsistencies Present: No\n\n"

                    "Output:\n"
                    "Inconsistencies detected: Yes\n"
                    "1. The chatbot incorrectly identifies the Right to Consumer Awareness as a new right introduced by the 2019 Act, even though it was already recognized under the Consumer Protection Act 1986.\n"
                    "2. The chatbot claims that consumers must physically visit consumer courts to file complaints, overlooking the provision in the 2019 Act allowing for e-filing of complaints.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The chatbot provides outdated pecuniary jurisdiction limits for district consumer courts, which were revised under the Consumer Protection Act 2019.\n"
                    "Turn 2\n"
                    "Inconsistencies Present: No\n"
                    "Turn 3\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The response mentions that only manufacturers are liable under product liability provisions, omitting sellers and service providers who are also included in the Act.\n\n"

                    "Output:\n"
                    "Inconsistencies detected: Yes\n"
                    "Example 4:\n"
                    "1. The chatbot provides outdated pecuniary jurisdiction limits for district consumer courts, which were revised under the Consumer Protection Act 2019.\n"
                    "2. The chatbot states that only manufacturers are liable under product liability provisions, omitting the inclusion of sellers and service providers under the Act.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: No\n"
                    "Turn 2\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The chatbot states that the Central Consumer Protection Authority (CCPA) can only handle individual complaints, which is incorrect since the CCPA can take suo-moto actions and initiate class actions.\n"
                    "Turn 3\n"
                    "Inconsistencies Present: No\n\n"

                    "Output:\n"
                    "Inconsistencies detected: Yes\n"
                    "1. The chatbot states that the Central Consumer Protection Authority (CCPA) can only handle individual complaints, failing to acknowledge its ability to take suo-moto actions and initiate class actions.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: No\n"
                    "Turn 2\n"
                    "Inconsistencies Present: No\n"
                    "Turn 3\n"
                    "Inconsistencies Present: No\n\n"

                    "Output:\n"
                    "Inconsistencies detected: No\n"
                    "No inconsistencies are present.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The chatbot claims that the Consumer Protection Act 2019 introduced a new right to seek redressal, but this right was also present in the earlier Act of 1986.\n"
                    "Turn 2\n"
                    "Inconsistencies Present: No\n"
                    "Turn 3\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The chatbot incorrectly states that mediation is mandatory for all consumer disputes under the 2019 Act, while it is an alternative, voluntary dispute resolution mechanism.\n\n"

                    "Output:\n"
                    "Inconsistencies detected: Yes\n"
                    "1. The chatbot mistakenly claims that the right to seek redressal is a new right under the 2019 Act, even though this right existed under the 1986 Act.\n"
                    "2. The chatbot incorrectly states that mediation is mandatory for all consumer disputes under the 2019 Act, while it is actually an alternative, voluntary dispute resolution mechanism.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The chatbot mentions the applicability of the Consumer Protection Act 2019 to the entire country, but it fails to acknowledge the special provisions for Jammu & Kashmir after the abrogation of Article 370.\n"
                    "Turn 2\n"
                    "Inconsistencies Present: No\n\n"

                    "Output:\n"
                    "Inconsistencies detected: Yes\n"
                    "1. The chatbot mentions the applicability of the Consumer Protection Act 2019 to the entire country but fails to acknowledge the special provisions for Jammu & Kashmir after the abrogation of Article 370.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: No\n"
                    "Turn 2\n"
                    "Inconsistencies Present: No\n"
                    "Turn 3\n"
                    "Inconsistencies Present: No\n\n"

                    "Output:\n"
                    "Inconsistencies detected: No\n"
                    "No inconsistencies are present.\n"
                    "<|endoftext|>\n\n"

                    "Input:\n"
                    "Chat History:\n"
                    "Turn 1\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The chatbot claims that an online complaint can only be filed through the National Consumer Helpline, but the Act allows for e-filing through all District, State, and National Commissions directly.\n"
                    "Turn 2\n"
                    "Inconsistencies Present: Yes\n"
                    "Inconsistencies:\n"
                    "1. The chatbot states that only consumer protection councils are responsible for handling complaints, ignoring the role of District, State, and National Commissions under the 2019 Act.\n\n"

                    "Output:\n"
                    "Inconsistencies detected: Yes\n"
                    "1. The chatbot claims that online complaints can only be filed through the National Consumer Helpline, ignoring the provision for e-filing directly through District, State, and National Commissions.\n"
                    "2. The chatbot states that only consumer protection councils are responsible for handling complaints, overlooking the roles of District, State, and National Commissions under the 2019 Act.\n"
                    "<|endoftext|>\n\n"

                    "Input:"
                    "Chat History: \n {analysis}"
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
print("START")
# Example query
first = True
folder_path = "./hall_detect/rag"
response_analysis_chain = create_rag_qa_chain_with_response_analysis(folder_path)

chat = '''
AI: Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you.


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


Is there anything else I can help you with?'''
# In 5th (start 0) chat, there is Inconsistencies with the karnataka drc number and email id.
# chat_sequence = process_chat_sequence(chat)
# res = response_analysis_chain.invoke({"chat":chat_sequence})
# print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
# print("Analysis-")
# print(res['analysis'])
# print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
# print("Result-")
# print(res['result'])
# print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
# print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")

chat = '''AI: Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you.


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
'''
chat_sequence = process_chat_sequence(chat)
res = response_analysis_chain.invoke({"chat":chat_sequence})
print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
print("Analysis-")
print(res['analysis'])
print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
print("Result-")
print(res['result'])