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

# Split the document into smaller chunks
chunks = re.split(r"(?:\n\n|\n\s*\n)", document)

# Summarize each chunk using Hugging Face model
summaries = []
for chunk in chunks:
    try:
        relevant_inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(relevant_inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary_hf = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary_hf)
    except:
        pass

# Combine the summaries into a single summary
summary = " ".join(summaries)

# Print or use the summary
print(summary)