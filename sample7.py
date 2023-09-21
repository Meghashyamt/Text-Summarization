import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from summarizer import Summarizer, TransformerSummarizer
from PyPDF2 import PdfReader
import re

# Load a pretrained summarization model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define the path to your PDF document
pdf_document_path = r"C:\Users\M_Thiruveedula\Downloads\Solution_python\Solution_python\python_solution\RFP-214.pdf"  # Replace with the actual path

# Read text from the PDF document
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
# ...

# Define a search function to find and extract content based on the keyword
def search_and_extract_content(keyword, text):
    paragraphs = text.split('\n')
    relevant_content = ""
    found_keyword = False
    
    for paragraph in paragraphs:
        if keyword.lower() in paragraph.lower():
            found_keyword = True
        elif found_keyword:
            relevant_content += paragraph + "\n"
            # Count the words in relevant content
            word_count = len(re.findall(r'\w+', relevant_content))
            if word_count >= 300:
                break  # Stop after the next 500 words following the keyword
        
    return relevant_content

# Perform the keyword-based search and content extraction
search_result = dataset.map(lambda example: {"relevant_content": search_and_extract_content(search_keyword, example["text"])})

# Extract the relevant content (next 500 words)
relevant_content = search_result["relevant_content"][0]

# Log the extracted content
logging.info("Extracted Content:\n%s", relevant_content)

# Print or use the extracted content
print("Extracted Content (Next 300 Words):")
print(relevant_content)

# Summarize the extracted content using Hugging Face model
if relevant_content:
    relevant_inputs = tokenizer(relevant_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(relevant_inputs["input_ids"], max_length=1500, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_hf = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
else:
    summary_hf = "No relevant content found."

# Print or use the Hugging Face summary as needed
print("Hugging Face Summary:")
print(summary_hf)

summarizer = pipeline("summarization")


# Generate a summary
summary = summarizer(relevant_content, max_length=200, min_length=30, do_sample=False)

# Print the summary
print(summary[0]['summary_text'])
