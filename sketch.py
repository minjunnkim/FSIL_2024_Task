
import os
import re
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import string
import xml.etree.ElementTree as ET

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from unstructured.documents.html import HTMLDocument

# Function to extract text from XBRL formatted as XML within a text file
def extract_text_from_xbrl(file_path):
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Parse the XML
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return ""
    
    # Namespace for finding tags might need to be adjusted based on the XBRL file
    ns = {'xbrl': 'http://www.xbrl.org/2003/instance'}
    # Extract narrative text using BeautifulSoup to handle HTML within XML tags
    text_blocks = []
    for non_num in root.findall('.//xbrl:nonNumeric', namespaces=ns):
        soup = BeautifulSoup(non_num.text, 'html.parser')
        text_blocks.append(soup.get_text())
    
    text = ' '.join(text_blocks)
    return text

# Function to extract the "Risk Factors" section from the extracted text
def extract_risk_factors_section(text):
    start_keyword = "Item 1A. Risk Factors"
    end_keyword = "Item 1B."
    start_idx = text.find(start_keyword)
    end_idx = text.find(end_keyword, start_idx + 1)
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx]
    else:
        return None

# Function to tokenize text, remove stopwords and punctuation, and count word frequency
def analyze_risk_factors(text):
    stop_words = set(stopwords.words('english'))  # Set of English stopwords
    words = word_tokenize(text.lower())  # Tokenize and convert to lower case
    filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return Counter(filtered_words)

# Visualization function
def plot_top_risks(counter, company_name, num_words=10):
    most_common = counter.most_common(num_words)
    words, counts = zip(*most_common)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.title(f'Top {num_words} Words in Risk Factors for {company_name}')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

# Process each company's filings
risk_factors_analysis = {}
for company in companies:
    company_path = os.path.join("sec-edgar-filings", company, "10-K")
    company_risks = Counter()  # Initialize counter for each company
    for root, dirs, files in os.walk(company_path):
        for file in files:
            if file=='full-submission.txt':
                file_path = os.path.join(root, file)
                raw_text = extract_text_from_xbrl(file_path)
                risk_section = extract_risk_factors_section(raw_text)
                if risk_section:
                    risks = analyze_risk_factors(risk_section)
                    company_risks += risks
    risk_factors_analysis[company] = company_risks
    plot_top_risks(company_risks, company)