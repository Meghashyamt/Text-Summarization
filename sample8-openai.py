import openai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = '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'

# Text to be summarized
input_text = """request and responses thereto shall be made in writing or by
email. If a Bidder accepts to extend the validity,  the Bid
Security shall also be suitably extended. A Bidder may
refuse the request without forfeiting its Bid Security. A  41
12.8.2 Bid is rejected for existence of conflict of interest, or more  42
Section 3. Instructions to Bidders and Bid Data Sheet
RFP for Appointment of Advanced Metering Infrastructure (AMI) Service Provider for Smart Prepaid Metering in  Punjab"""

# Define the prompt for GPT-3.5
prompt = f"Summarize the following text:\n{input_text}\n\nSummary:"

# Maximum length of the generated summary
max_summary_length = 150  # You can adjust this as needed

# Call the OpenAI API to generate the summary
response = openai.Completion.create(
    engine="gpt-3.5-turbo",
    prompt=prompt,
    max_tokens=max_summary_length,
    api_key=api_key
)

# Extract the generated summary from the response
summary = response.choices[0].text.strip()

# Print the summarized text
print(summary)
