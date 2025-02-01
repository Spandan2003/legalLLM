import os
from PyPDF2 import PdfReader
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def disp_dict(arr):
    print("Start Dictionary")
    for k,v in arr.items():
        print("-----------------------------------------------------")
        print(k)
        print(v)
        print("-----------------------------------------------------")
    print("End Dictionart")


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
    llm_model = "microsoft/phi-2"
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
        llm_pipeline = pipeline(
            "text-generation",
            model=llm_model,
            device=0 if device == "cuda" else -1,
            return_full_text=False,
            max_new_tokens=1024,
            temperature=0.5,
            do_sample=True
        )
        return HuggingFacePipeline(pipeline=llm_pipeline)

    # Step 5: Create a chain for analyzing responses
    def get_response_analysis_chain(retriever, llm):
        # Prompts for contextualizing and analyzing
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = PromptTemplate(
            template=contextualize_q_system_prompt,
            input_variables=["history", "query"]
        )
        reformulate_chain = contextualize_q_prompt | llm

        # analysis_prompt_template = PromptTemplate(
        #     template=(
        #         "You are an evaluator reviewing the response of a chatbot. "
        #         "Based on the provided context, history, and query, identify inconsistencies or falsehoods. "
        #         "Use the FactScore model to analyze facts in the response.\n\n"
        #         "Context: {context}\n\n"
        #         "History: {history}\n\n"
        #         "Query: {query}\n\n"
        #         "Response: {response}\n\n"
        #         "Fact Classifications: {fact_classifications}\n\n"
        #         "Based on the above, provide a detailed assessment of the response."
        #     ),
        #     input_variables=["context", "history", "query", "response", "fact_classifications"]
        # )
        # analysis_chain = analysis_prompt_template | llm

        # FactScore decomposition and classification
        decompose_facts_prompt = PromptTemplate(
            template="Given the response: {response}, break it into independent facts. Each fact should be on a new line. Do not repeat a fact.",
            input_variables=["response"]
        )
        decompose_facts_chain = decompose_facts_prompt | llm

        classification_prompt = PromptTemplate(
            template=(
                "Classify each fact based on the context without giving any explaination. Just use the code words to show your classification:\n\n"
                "Context: {context}\n\n"
                "Query: {reformulated_query}\n\n"
                "Facts: \n{fact}\n\n"
                "Label each fact in Facts as one of code words 'Supported', 'Not Supported', or 'Irrelevant'. "
            ),
            input_variables=["evidence", "fact"]
        )
        classification_chain = classification_prompt | llm

        # Helper function for fact classification
        def classify_facts(facts, context, history, reformulated_query):
            batched_facts = [facts[i:i + 10] for i in range(0, len(facts), 10)]
            all_classifications = []
            for batch in batched_facts:
                facts_text = "\n".join([f"Fact: {fact}" for fact in batch])
                response = classification_chain.invoke({"context": context, "fact": facts_text, "history":history, "reformulated_query":reformulated_query})
                all_classifications.extend(response.split("\n"))  # Process response into individual classifications
            return all_classifications

        # Define the combined chain
        class ResponseAnalysisChain:
            def __init__(self, retriever, reformulate_chain, decompose_facts_chain, classification_chain):
                self.retriever = retriever
                self.reformulate_chain = reformulate_chain
                self.analysis_chain = analysis_chain
                self.decompose_facts_chain = decompose_facts_chain
                self.classification_chain = classification_chain

            def __call__(self, inputs):
                # Extract inputs
                query = inputs["query"]
                history = inputs["history"]
                response = inputs["response"]

                # Reformulate the query
                reformulated_query = self.reformulate_chain.invoke({"history": history, "query": query})

                # Retrieve relevant context
                context_docs = self.retriever.get_relevant_documents(reformulated_query)
                context = " ".join([doc.page_content for doc in context_docs])

                # Decompose response into atomic facts
                decomposed_facts = self.decompose_facts_chain.invoke({"response": response})
                atomic_facts = decomposed_facts.split("\n")  # Convert decomposed facts into a list

                # Classify each fact based on the context
                fact_classifications = classify_facts(atomic_facts, context, history, reformulated_query)

                # Pass everything to the analysis chain
                final_inputs = {
                    "context": context,
                    "history": history,
                    "query": query,
                    "response": response,
                    "classification": fact_classifications,
                }
                return final_inputs

        return ResponseAnalysisChain(retriever, reformulate_chain, decompose_facts_chain, classification_chain)


    # Build the pipeline
    text = get_pdf_text(folder_path)
    text_chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    response_analysis_chain = get_response_analysis_chain(retriever, llm)

    return response_analysis_chain

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
    return [element.strip() for element in elements if element.strip()]


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
        if elements[i].startswith("AI:") and i>0:
            query = elements[i-1].replace("Human:", "").strip()
            history = "\n".join(elements[:i-1]).strip()
            response = elements[i].replace("AI:", "").strip()
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
# In 5th (start 0) chat, there is hallucination with the karnataka drc number and email id.
chat_sequence = process_chat_sequence(chat)
folder_path = "./hall_detect/rag"

# Create the RAG-based QA chain
print("START")
# Example query
response_analysis_chain = create_rag_qa_chain_with_response_analysis(folder_path)
print(chat_sequence[5])
analysis = response_analysis_chain(chat_sequence[5])


count = {'t':0, 's':0, 'ns':0, 'ir':0, 'others':0}
for facts in analysis["classification"]:
    for fact in facts.split():
        count['t'] += 1  # Increment total count for every fact
        if "not supported" in fact.lower():
            count['ns'] += 1
        elif "supported" in fact.lower():
            count['s'] += 1
        elif "irrelevant" in fact.lower():
            count['ir'] += 1
        else:
            count['others'] += 1

# Print the counts
disp_dict(analysis)
print(count)
print("FACTSCORE = ",count['s']/count['t'])