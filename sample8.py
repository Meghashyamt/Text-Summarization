import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from summarizer import Summarizer
from PyPDF2 import PdfReader
import re

# Load a pretrained summarization model
model_name = "t5-small"
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

# Define a function to extract content before and after the keyword, limited to 300 words in each direction
def extract_content_around_keyword(document, keyword, max_words=300):
    keyword_positions = [match.start() for match in re.finditer(re.escape(keyword), document, re.IGNORECASE)]
    extracted_content = ""
    
    for position in keyword_positions:
        start = max(0, position - max_words)
        end = min(len(document), position + max_words)
        extracted_content += document[start:end]
    
    return extracted_content

# Extract content around the keyword
relevant_content = extract_content_around_keyword(document, search_keyword)

# Log the extracted content
logging.info("Extracted Content Around Keyword:\n%s", relevant_content)

# Print or use the extracted content
print("Extracted Content Around Keyword (Up to 300 Words):")
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

# Summarize the extracted content using the 'summarizer' library
summarizer_model = Summarizer()
summarizer_summary = summarizer_model(relevant_content)

# Print or use the 'summarizer' summary as needed
print("Summarizer Library Summary:")
print(summarizer_summary)

summarizer = pipeline("summarization")


# Generate a summary
summary = summarizer(relevant_content, max_length=200, min_length=30, do_sample=False)
print("Summarization")
# Print the summary
print(summary[0]['summary_text'])