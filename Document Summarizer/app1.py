from flask import Flask, render_template, request
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from summarizer import Summarizer
from PyPDF2 import PdfReader
import re
import spacy
import time

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Load a pretrained summarization model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        keyword = request.form["keyword"]

        # Inside your `index` route function
        if uploaded_file and keyword:
            # Measure the start time for content extraction
            extraction_start_time = time.time()

            # Save the uploaded file
            uploaded_file.save("uploaded_file.pdf")

            # Extract text from the PDF document
            document = extract_text_from_pdf("uploaded_file.pdf")

            # Measure the end time for content extraction
            extraction_end_time = time.time()

            # Calculate the time taken for content extraction
            extraction_time = extraction_end_time - extraction_start_time

            # Count words in the extracted content
            word_count = len(document.split())

            # Extract content around the keyword
            relevant_content = extract_content_around_keyword(document, keyword)

            # Measure the start time for summarization
            summarization_start_time = time.time()

            # Summarize the extracted content using Hugging Face model
            if relevant_content:
                relevant_inputs = tokenizer(relevant_content, return_tensors="pt", max_length=1024, truncation=True)
                summary_ids = model.generate(relevant_inputs["input_ids"], max_length=1500, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
                summary_hf = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            else:
                summary_hf = "No relevant content found."

            # Measure the end time for summarization
            summarization_end_time = time.time()

            # Calculate the time taken for summarization
            summarization_time = summarization_end_time - summarization_start_time

            # Summarize the extracted content using the 'summarizer' library
            summarizer_model = Summarizer()
            summarizer_start_time = time.time()
            summarizer_summary = summarizer_model(relevant_content)
            summarizer_end_time = time.time()
            summarizer_time = summarizer_end_time - summarizer_start_time

            summarizer = pipeline("summarization", model="google/pegasus-xsum")
            

            return render_template("index.html", keyword=keyword, extracted_content=relevant_content, hf_summary=summary_hf,
                                summarizer_summary=summarizer_summary, spacy_summary=summarizer, extraction_time=extraction_time,
                                word_count=word_count, summarization_time=summarization_time, summarizer_time=summarizer_time)


    return render_template("index.html", keyword="", extracted_content="", hf_summary="", summarizer_summary="")

#if __name__ == "__main__":
#    app.run(debug=True)
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_content_around_keyword(document, keyword, max_words=300):
    keyword_positions = [match.start() for match in re.finditer(re.escape(keyword), document, re.IGNORECASE)]
    extracted_content = ""

    for position in keyword_positions:
        start = max(0, position - max_words)
        end = min(len(document), position + max_words)
        extracted_content += document[start:end]
    
    extracted_content = re.sub(r"[^a-zA-Z0-9\s]", "", extracted_content)

    extracted_content = re.sub(r"\d+", "", extracted_content)
    extracted_content = extracted_content.replace("section", "")
    
    return extracted_content

if __name__ == "__main__":
    app.run(debug=True)
