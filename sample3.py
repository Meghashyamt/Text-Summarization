import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
from summarizer import Summarizer,TransformerSummarizer
from PyPDF2 import PdfReader
import re

# Load a pretrained summarization model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define the path to your PDF document
#pdf_document_path = r"C:\Users\M_Thiruveedula\Downloads\Solution_python\Solution_python\python_solution\Contract\Sponsorship\HFI- Calcutta Sponsorship Agreement_23 Sept 2021.docx.pdf" # Replace with the actual path
pdf_document_path=r"C:\Users\M_Thiruveedula\Downloads\Solution_python\Solution_python\python_solution\RFP-214.pdf"
# Read text from the PDF document
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     pdf_reader = PdfReader(open(pdf_path, "rb"))
#     for page in range(pdf_reader.numPages):
#         text += pdf_reader.getPage(page).extractText()
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from the PDF document
document = extract_text_from_pdf(pdf_document_path)

# Ask for the search keyword
search_keyword = input("Enter the keyword to search for in the document: ")

# Tokenize the entire document
inputs = tokenizer(document, return_tensors="pt", max_length=1024, truncation=True)

# Use the 'nlp' library for keyword-based search
# Load a dataset with your document for easy search
from nlp import Dataset
dataset = Dataset.from_dict({"text": [document]})

# Define a search function to find and extract content based on the keyword
def search_and_extract_content(keyword, text):
    paragraphs = text.split('\n')
    relevant_content = ""
    found_keyword = False
    
    for paragraph in paragraphs:
        if keyword.lower() in paragraph.lower():
            found_keyword = True
            relevant_content += paragraph + "\n"
        elif found_keyword:
            break  # Stop adding paragraphs once a new section starts
        
    return relevant_content

# Perform the keyword-based search and content extraction
search_result = dataset.map(lambda example: {"relevant_content": search_and_extract_content(search_keyword, example["text"])})
print(search_result)

# Extract the relevant content
relevant_content = search_result["relevant_content"][0]
print(relevant_content)

# Summarize the relevant content
if relevant_content:
    relevant_inputs = tokenizer(relevant_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(relevant_inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
else:
    summary = "No relevant content found."

# Print or use the summary as needed
print(summary)

# ### HUGGING FACE SUMMARIZER
print("Hugging face")
summarizer = pipeline("summarization")
#ARTICLE="This Agreement shall be construed in accordance with, and governed by, the laws of India. The courts at Gurgaon shall have exclusive jurisdiction over all matters arising out of or in relation to this Agreement."
ARTICLE="consumer indexing will be carried out/verified for the incoming population of smart meters for end-to end metering at contiguous electrical locations in the selected AMI Project Area only. The responsibility for consumer indexing for dispersed metering at non-contiguous electrical locations in the selected AMI Project Area shall also lie with the AMISP/Bidder. For this a door-to-door survey shall be required for each meter installed and tallying it with the consumer related records (physical, electrical and commercial) available with the Utility. In establishing the linkage of the consumer meter to the electric network, the asset (including the meter) codification as used by the utility GIS (or as per standards set by the utility) shall be strictly followed."
output=summarizer(summary, min_length=10, do_sample=False)
print(output)


###BERT MODEL
print("BERT")

bert_model = Summarizer()
bert_summary = ''.join(bert_model(summary, min_length=10))
print(bert_summary)


### GPT2 MODEL
print("GPT2")

GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(summary, min_length=10))
print(full)

# import logging
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
# from summarizer import Summarizer,TransformerSummarizer
# from PyPDF2 import PdfReader
# import re

# # Configure logging
# log_file = "summary_logs.txt"
# logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# # Load a pretrained summarization model
# model_name = "facebook/bart-large-cnn"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Define the path to your PDF document
# pdf_document_path = r"C:\Users\M_Thiruveedula\Downloads\Solution_python\Solution_python\python_solution\Contract\Sponsorship\HFI- Calcutta Sponsorship Agreement_23 Sept 2021.docx.pdf"  # Replace with the actual path

# # Read text from the PDF document
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     pdf_reader = PdfReader(pdf_path)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# # Extract text from the PDF document
# document = extract_text_from_pdf(pdf_document_path)

# # Ask for the search keyword
# search_keyword = input("Enter the keyword to search for in the document: ")

# # Log the input keyword
# logging.info("Input Keyword: %s", search_keyword)

# # Tokenize the entire document
# inputs = tokenizer(document, return_tensors="pt", max_length=1024, truncation=True)

# # Use the 'nlp' library for keyword-based search
# # Load a dataset with your document for easy search
# from nlp import Dataset
# dataset = Dataset.from_dict({"text": [document]})

# # Define a search function to find and extract content based on the keyword
# def search_and_extract_content(keyword, text):
#     paragraphs = text.split('\n')
#     relevant_content = ""
#     found_keyword = False
    
#     for paragraph in paragraphs:
#         if keyword.lower() in paragraph.lower():
#             found_keyword = True
#         elif found_keyword:
#             relevant_content += paragraph + "\n"
        
#     return relevant_content

# # Perform the keyword-based search and content extraction
# search_result = dataset.map(lambda example: {"relevant_content": search_and_extract_content(search_keyword, example["text"])})

# # Extract the relevant content
# relevant_content = search_result["relevant_content"][0]

# # Log the output content
# logging.info("Output Content:\n%s", relevant_content)

# # Summarize the relevant content
# if relevant_content:
#     relevant_inputs = tokenizer(relevant_content, return_tensors="pt", max_length=1024, truncation=True)
#     summary_ids = model.generate(relevant_inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# else:
#     summary = "No relevant content found."

# # Log the summary
# logging.info("Summary:\n%s", summary)

# # Print or use the summary as needed
# print(summary)

# # ### HUGGING FACE SUMMARIZER
# print("Hugging face")
# summarizer = pipeline("summarization")
# #ARTICLE="This Agreement shall be construed in accordance with, and governed by, the laws of India. The courts at Gurgaon shall have exclusive jurisdiction over all matters arising out of or in relation to this Agreement."
# ARTICLE="consumer indexing will be carried out/verified for the incoming population of smart meters for end-to end metering at contiguous electrical locations in the selected AMI Project Area only. The responsibility for consumer indexing for dispersed metering at non-contiguous electrical locations in the selected AMI Project Area shall also lie with the AMISP/Bidder. For this a door-to-door survey shall be required for each meter installed and tallying it with the consumer related records (physical, electrical and commercial) available with the Utility. In establishing the linkage of the consumer meter to the electric network, the asset (including the meter) codification as used by the utility GIS (or as per standards set by the utility) shall be strictly followed."
# output=summarizer(ARTICLE, min_length=10, do_sample=False)
# print(output)


# ###BERT MODEL
# print("BERT")

# bert_model = Summarizer()
# bert_summary = ''.join(bert_model(ARTICLE, min_length=10))
# print(bert_summary)


# ### GPT2 MODEL
# print("GPT2")

# GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
# full = ''.join(GPT2_model(ARTICLE, min_length=10))
# print(full)

### XLNet MODEL
# print("XLNet")

# model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
# full = ''.join(model(ARTICLE, min_length=60))
# print(full)
