from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nlp
from PyPDF2 import PdfReader

# Load a pretrained summarization model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define the path to your PDF document
pdf_document_path = r"C:\Users\M_Thiruveedula\Downloads\Solution_python\Solution_python\python_solution\Contract\Sponsorship\HFI- Calcutta Sponsorship Agreement_23 Sept 2021.docx.pdf" # Replace with the actual path

# Read text from the PDF document
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from the PDF document
document = extract_text_from_pdf(pdf_document_path)

# Define your search keyword
search_keyword = "Transparency"

# Tokenize the entire document
inputs = tokenizer(document, return_tensors="pt", max_length=1024, truncation=True)

# Use the 'nlp' library for keyword-based search
# Load a dataset with your document for easy search
dataset = nlp.Dataset.from_dict({"text": [document]})

# Define a search function to find and extract content based on the keyword
def search_and_extract_content(keyword, text):
    sentences = text.split('.')  # Split the document into sentences
    relevant_sentences = []
    
    for sentence in sentences:
        if keyword in sentence:
            relevant_sentences.append(sentence.strip())
    
    # Capture a context window around the keyword
    context_window = 2  # You can adjust this window size
    start_index = max(0, relevant_sentences.index(sentence) - context_window)
    end_index = min(len(relevant_sentences), relevant_sentences.index(sentence) + context_window + 1)
    context = " ".join(relevant_sentences[start_index:end_index])
    
    return context

# Perform the keyword-based search and content extraction
search_result = dataset.map(lambda example: {"relevant_content": search_and_extract_content(search_keyword, example["text"])})

# Extract the relevant content
relevant_content = search_result["relevant_content"][0]

# Summarize the relevant content
if relevant_content:
    relevant_inputs = tokenizer(relevant_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(relevant_inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
else:
    summary = "No relevant content found."

# Print or use the summary as needed
print(summary)
