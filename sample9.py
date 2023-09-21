
text_example="""17 as may be extended from time to time. 12.1.1 All such offers, and terms and conditions set forth in this RFP shall be valid for the AMISP till the successful completion of the Project. 12.1.2 In exceptional circumstance, Utility may solicit the Bidder‘s consent to an extension of the Bid validity period. The request and responses thereto shall be made in writing or by email. If a Bidder accepts to extend the validity, the Bid Security shall also be suitably extended. A Bidder may refuse the request without forfeiting its Bid Security. A 41 Section 3. Instructions to Bidde"""
from transformers import pipeline
summarizer = pipeline("summarization", model = "google/pegasus-xsum")
summarizer(text_example)