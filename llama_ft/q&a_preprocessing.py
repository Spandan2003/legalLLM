import os
import re
from docx import Document
import pandas as pd


# Get the current directory
current_directory = os.getcwd()
file_path = r"./code/legalLLM/llama_ft/dataset/"






def formating(text):
    user = ["User:", "Aggrieved User:", "Sura:", "Aggrieved Consumer:",  "Client:", "U:", 
            "Consumer- Consumer:", "User-", "Aggrieved Consumer-", "Consumer-", "Consumer:", "User :", "Userser:", "User." ]
    chatbot = ["Chatbot:", "The Chatbot:", "Chatot:", "Lawyer:", "C:", "Chatbot :", "Chatbot-", "The Chatbox:", "The chatbox:", 
               "Chatbor:", "Chatbout:"]
    for el in user:
        if(text.find(el)==0):
            text = text.replace(el, "User:", 1)
            break
    for el in chatbot:
        if(text.find(el)==0):
            text = text.replace(el, "Chatbot:", 1)
            break
    return text

def docx_to_txt(docx_dir, txt_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(txt_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(docx_dir):
        if filename.endswith(".docx"):
            # Construct the input and output file paths
            docx_filepath = os.path.join(docx_dir, filename)
            output_filepath = os.path.join(txt_dir, os.path.splitext(filename)[0] + ".txt")

            # Read content from the docx file
            doc = Document(docx_filepath)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            content = formating(content)

            # Write content to the txt file
            with open(output_filepath, "w") as f:
                f.write(content)

            print(f"Converted {filename} to {os.path.splitext(filename)[0]}.txt")


def refine_txt(txt_dir, llm_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(llm_dir, exist_ok=True)

    # Iterate through each file in the directory
    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(txt_dir, filename)
            output_file_path = os.path.join(llm_dir, filename)

            with open(input_file_path, "r", encoding="utf-8") as input_file:
                with open(output_file_path, "w+", encoding="utf-8") as output_file:
                    first_line = True
                    for line in input_file:
                        # Replace "User:" with "</s> <s><INST>"
                        line = formating(line) 
                        if(line==None or (line.find("User:")==-1 and first_line)):
                            continue
                        if first_line:
                            line = line.replace("User:", "<s><INST>")
                            first_line = not first_line
                        else:
                            line = line.replace("User:", "</s> <s><INST>")
                        # Replace "Chatbot:" with "</INST>"
                        line = line.replace("Chatbot:", "</INST>")

                        # Write modified line to output file
                        #print(line)
                        output_file.write(line)
                    output_file.write("</s>")
    print("Replacement completed successfully.")

def combine_all_txt(txt_dir, output_file):
    lines = []
    for filename in os.listdir(txt_dir):
        with open(os.path.join(txt_dir, filename), "r", encoding="utf-8") as input_file:
            for line in input_file:
                lines.append(line)

    with open(file_path + output_file, "w+") as output_file:
        for line in lines:
            output_file.write(line)

def select_lines(txt_dir, query, output_file):
    lines = []
    for filename in os.listdir(txt_dir):
        with open(os.path.join(txt_dir, filename), "r", encoding="utf-8") as input_file:
            for line in input_file:
                lines.append(line)
    with open(file_path + output_file, "w+") as output_file:
        for line in lines:
            ind = line.find(query)
            if(ind==-1):
                continue
            output_file.write(line[:ind+1])
            output_file.write('\n')




# Define input and output directories
docx_dir = file_path + "Q&A"
txt_dir = file_path + "Q&A_txt"
llm_dir = file_path + "Q&A_llm"


# docx_to_txt(docx_dir, txt_dir)      #Convert all files of .docx format to .txt format


# combine_all_txt(txt_dir, "delete_asap2.txt")    # Combines all txt files into a single large txt files
# select_lines(txt_dir, ':', "delete_asap2.txt")  # Stores all lines containing the given query phrase into a single txt file


# refine_txt(txt_dir, llm_dir)            # Convert all chats into the format required by Llama


text = """
<s><INST>  I have an issue with regards to my insurance policy. How do I resolve it.

</INST> Could you please elaborate the issue.

</s> <s><INST>  My husband is an employee at Rashtriya Ispat Nigam Ltd., (RINL). The RINL in order to safeguard the interests of the employees obtained Group Personal Accident Insurance Policies New India Assurance Company Limited and for the same RINL used to deduct a certain amount from my husband’s salary to pay the premium for several years. In 2016, My husband fell down from a two wheeler and sustained a head injury. He was immediately admitted in the hospital, where he underwent the treatment. He was treated in different hospitals for the past two years and he eventually died. When I went to claim insurance it was denied. 

</INST> I’m so sorry this happened to you. Could you please mention when did this incident happen

</s> <s><INST>  It happened on 05/04/2016.

</INST> could you please mention further details such as employment details of your husband such as what is his post, what is his salary and of the insurance policy, specifically the amount, policy number, the period of policy.

</s> <s><INST>  My Husband worked in RINL as a Senior Foreman, CMM Section with Employee No. 11379. His salary was Rs.50,000. I do not know the premium amount deducted from his salary. But the coverage is for Rs.31,00,000 and the policy commences from 1st april to 31st march of every financial year. Thus considering the period 01/04/2016 to 31/03/2017. The policy number is 62030042160100000008. 

</INST> Thank you for sharing those details. Could you please further mention if you have reported the incident to police or got an FIR registered. If so, could you provide those details.

</s>
"""

def single_turn(folder_path, output_path):
    # Extract user inputs and responses using regular expressions
    dataset = pd.DataFrame({'prompt':[], 'response':[]})
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as input_file:
            text = input_file.read()
            input_output_pairs = re.findall(r'<INST>(.*?)</INST>\s+(.*?)</s>', text, re.DOTALL)
        dataset_additional = pd.DataFrame(input_output_pairs, columns=['prompt', 'response'])
        dataset = pd.concat([dataset, dataset_additional])
    dataset.reset_index(drop=True, inplace=True)
    dataset.to_csv(output_path)

def multi_turn(folder_path, output_path):
    # Extract user inputs and responses using regular expressions
    dataset = pd.DataFrame({'prompt':[], 'response':[]})
    for filename in os.listdir(folder_path):
        context = ""
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as input_file:
            text = input_file.read()
            input_output_pairs = re.findall(r'<INST>(.*?)</INST>\s+(.*?)</s>', text, re.DOTALL)
        new_pairs = []
        for inp, out in input_output_pairs:
            context = context + inp
            new_pairs.append((context, out))
            context = context + out
        dataset_additional = pd.DataFrame(new_pairs, columns=['prompt', 'response'])
        dataset_additional
        dataset = pd.concat([dataset, dataset_additional])
    dataset.reset_index(drop=True, inplace=True)
    return dataset

# single_turn(llm_dir, file_path + "Q&A-singleturn.csv")
multi_turn(llm_dir, None)



      