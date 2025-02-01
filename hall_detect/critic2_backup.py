import os
from PyPDF2 import PdfReader
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
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
            device=3,  # Use GPU if available
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
                "You are an evaluator in an AI company whose job is to determine the quality of a legal chatbot which is to give correct and flawless legal advice to the user. For this you are provided the context (the textual knowledge related to the query), History (the previous chat between the user and the chatbot), the Query (question or statement by the user on which the response is based) and the Response (by the chatbot)\n"
                "Your goal is to review the response and point out the inconsistencies or falsehood in the chatbot. This is a list of the possible faults\n"
                "1. Any response which contradicts the context or any part of the context\n"
                "2. Any response that has information or facts which do not exist in the context\n"
                "3. Any response that has information like contact number or address or so on of a company, organization or government body that has not been mentioned in the context\n"
                "Your goal is to review the response and find out whether there are inconsistencies or falsehood in the chatbot.\n"
                "Only state the assumptions which are based upon false statements. If any assumption is based upon logic, do not mention it.\n"
                "Ensure that the response is concise, without repetition and ends with an `<|endoftext|>` token"
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

                "\nExample 6:\n"
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
                "1. The response claims that the Act mandates addressing workplace stress and mental health issues, which is not supported by the context.\n<|endoftext|>"

                "\nExample 7:\n"
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
                "1. The response incorrectly claims that GDPR allows individuals to claim compensation for breaches directly, which is not mentioned in the context.\n<|endoftext|>"

                "\nExample 8:\n"
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
                "Inconsistencies Present: No\n<|endoftext|>"

                "\nExample 9:\n"
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
                "1. The response incorrectly states that tenants may be entitled to compensation under the Housing Act 1988, which is not mentioned in the context.\n<|endoftext|>"

                "\nExample 10:\n"
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
                "Inconsistencies Present: No\n<|endoftext|>"

                "\nExample 11:\n"
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


                "Context:\n {context}\nNational Consumer Helpline (1800-11-4000) or UMANG App for more assistance\n"
                "History:\n {history}\n"
                "Query:\n {query}\n"
                "Response:\n {response}\n"
            ),
            input_variables=["context", "history", "query", "response"]
        )

        analysis_chain = analysis_prompt_template | llm2

        # Define a custom chain that combines reformulating and analysis
        class ResponseAnalysisChain:
            def __init__(self, retriever, reformulate_chain, analysis_chain):
                self.retriever = retriever
                self.reformulate_chain = reformulate_chain | StrOutputParser()
                self.analysis_chain = analysis_chain | StrOutputParser()


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
                context = " ".join([doc.page_content for doc in context_docs]).replace('/n', ' ')

                # Pass everything to the analysis chain
                final_inputs = {
                    "context": context,
                    "history": history,
                    "query": query,
                    "reformulated": reformulated_query,
                    "response": response,
                }
                global first
                if first==True:
                    # print(final_inputs['context'].replace('/n', ' '))
                    first = False
                return cutoff_at_stop_token(self.analysis_chain.invoke(final_inputs))

        return ResponseAnalysisChain(retriever, reformulate_chain, analysis_chain)

    # Build the pipeline
    text = get_pdf_text(folder_path)
    text_chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm1, llm2 = get_llm()
    response_analysis_chain = get_response_analysis_chain(retriever, llm1, llm2)
    #res = detect_hallucination_per_chat(chat_sequence, response_analysis_chain, llm1)
    return response_analysis_chain, llm1

def detect_hallucination_per_chat(chat, chain, llm):
    analysis = []
    for i, chat_triple in enumerate(chat):
        analysis.append("\nTurn "+ str(i+1)+ "\n" + chain(chat_triple))
    prompt = PromptTemplate(
            template=(
                # "You are given an individual history of Inconsistencies detected in a chatbot. You are an evaluator and your task is to use all of these statements and give out simple description answering whether Inconsistencies are present as a whole in the chatbot. Do not individually mention each Inconsistencies but give a single statement describing whether Inconsistencies are present or not. It should be in this format:"
                # "Inconsistencies are present"
                # "The folllowing information is inconsistent"
                # "\nAnswer in short. What is the final response to the questions: Are there any Inconsistencies in this chat? List the most significant of them"
                "You are provided with a record of inconsistencies detected during interactions with a chatbot across multiple turns. As an evaluator, your task is to analyze all the detected inconsistencies collectively and provide a concise assessment of whether inconsistencies are present in the chatbot’s responses as a whole."

                "Your response should not list individual inconsistencies but instead deliver a unified summary. Use one of the following formats to present your evaluation:"

                "'Inconsistencies are present.'"
                "'The following information is inconsistent: [briefly describe the most significant inconsistencies].'"
                "Provide a short and clear final answer to these questions:"

                "Are there any inconsistencies in this chat?"
                "What are the most significant inconsistencies observed?"
                "Example 1:\n"
                "Chat History:\n"
                "Turn 1\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The response mentions the toll-free numbers for the LPG distributors in Karnataka, but the context explicitly states that the Hindustan Petroleum toll-free number varies by region and should be checked on their website.\n"
                "2. The response provides the phone numbers for the Karnataka State Consumer Dispute Redressal Commission (KSDRC), but the context does not mention this commission or its contact details.\n"
                "\n"
                "Turn 2\n"
                "Inconsistencies Present: No\n"
                "\n"
                "Turn 3\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The response mentions the toll-free numbers for LPG distributors in Karnataka, but the context only mentions Indian Oil and Bharat Petroleum, not Hindustan Petroleum.\n"
                "2. The response provides a phone number for the Karnataka State Consumer Dispute Redressal Commission (KSDRC) that is not mentioned in the context, and also provides an email address that is not mentioned in the context.\n"
                "\n"
                "Output:\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. Responses mention toll-free numbers and organizations not specified or contextually incorrect, with variations across turns.\n<|endoftext|>"
                "\n"
                "Example 2:\n"
                "Chat History:\n"
                "Turn 1\n"
                "Inconsistencies Present: No\n"
                "\n"
                "Turn 2\n"
                "Inconsistencies Present: No\n"
                "\n"
                "Turn 3\n"
                "Inconsistencies Present: No\n"
                "\n"
                "Output:\n"
                "Inconsistencies Present: No\n<|endoftext|>"
                "\n"
                "Example 3:\n"
                "Chat History:\n"
                "Turn 1\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The chatbot references toll-free numbers for regional services without specifying their regional restrictions as required by the context.\n"
                "\n"
                "Turn 2\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The chatbot adds information about unrelated services that are not mentioned or relevant to the context.\n"
                "\n"
                "Turn 3\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The chatbot combines conflicting service details, leading to a confusing response.\n"
                "\n"
                "Output:\n"
                "Inconsistencies Present: Yes\n"
                "Inconsistencies:\n"
                "1. The chatbot consistently provides regionally incorrect or irrelevant details, and contradicts context requirements across multiple turns.\n<|endoftext|>\n"
                "Chat History: \n {analysis}"
            ),
            input_variables=["analysis"]
        )
    combination_chain = prompt | llm
    print("Analysis")
    for a in analysis:
        print(a)
    print("x--x-x-x-x-x-x-x-x-x-x-x-x-x")
    return combination_chain.invoke("".join(analysis))
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
chat_sequence = process_chat_sequence(chat)
folder_path = "./hall_detect/rag"

# Create the RAG-based QA chain
print("START")
# Example query
first = True
response_analysis_chain, llm = create_rag_qa_chain_with_response_analysis(folder_path)
# disp_dict(chat_sequence[5])
# print(chat_sequence)
for i in range(1):
    print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x")
    # res = response_analysis_chain(chat_sequence)
    res = detect_hallucination_per_chat(chat_sequence, response_analysis_chain, llm)
    print("----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----x----xspandan")
    print(res)