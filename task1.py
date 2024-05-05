"""
Financial Services Innovation Lab, Georgia Tech

Programming Task for Summer Research 2024

Minjun Andrew Kim
"""

#########################################################

"""
Task 1.1 - Download Data from SEC-EDGAR

Used sec_edgar_downloader to gather 10-K Filings for:

(MAMAA)
Meta Platforms, Inc. -- CIK 0001326801
Apple Inc.           -- CIK 0000320193
Microsoft Corp       -- CIK 0000789019
Amazon.com, Inc.     -- CIK 0001018724
Alphabet Inc.        -- CIK 0001652044

10-K Filings are stored in /sec-edgar-filings, under each company's folder
"""

from sec_edgar_downloader import Downloader

dl = Downloader("Minjun", "ckandrew04@gmail.com")

companies = ["META", "AAPL", "MSFT", "AMZN", "GOOGL"]

for company in companies:
    dl.get("10-K", company)

"""
Function for Task 2
"""

def download_10k(company):
    """
    Helper function that downloads the desired company ticker's 10-K Filings

    Args:
        company (String): company ticker
    """
    dl.get("10-K", company)

#########################################################

"""
Task 1.2 - Text Analysis

Used openai as LLM to perform text analysis on the 10-K Filings from 1.1
"""

"""
Data Cleaning

Since the full-submission.txt given from sec-edgar-downloader consists of mostly useless data, 
we only extract the first <DOCUMENT> element in the file, which includes all the 10-K Filing texts
"""

import os
from bs4 import BeautifulSoup
import re

items = ['item 1. ', 'item 1a. ', 'item 1b. ', 'item 1c. ', 'item 2. ', 
         'item 3. ', 'item 4. ', 'item 5. ', 'item 6. ', 'item 7. ', 'item 7a. ', 
         'item 8. ', 'item 9. ', 'item 9a. ', 'item 9b. ', 'item 9c. ', 'item 10. ', 
         'item 11. ', 'item 12. ', 'item 13. ', 'item 14. ', 'item 15. ', 'item 16. ']

def extract_10k_document(file_path):
    """
    Initially extraction from the full-submission.txt
    We only need the first <DOCUMENT> in the text file

    Args:
        file_path (String): path to full-submission.txt

    Returns:
        String: extracted document containing 10-K Filing contents
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    start_index = content.find('<DOCUMENT>')
    end_index = content.find('</DOCUMENT>', start_index) + len('</DOCUMENT>')

    # Extracting the content of the first <DOCUMENT> block
    document = content[start_index:end_index]

    return document

def clean_and_format_html(document):
    """
    Initially clean and format the html contents in the document file before organizing

    Args:
        document (String): original document text

    Returns:
        String: initially cleaned text
    """
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(document, 'lxml')

    # Remove script, style, and other unnecessary tags
    for script_or_style in soup(["script", "style", "img", "a"]):
        script_or_style.decompose()

    # Function to clean individual strings
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
        text = re.sub(r'\s+\.', '.', text)  # Remove spaces before periods
        return text.strip()

    # Extract and format text
    text = []
    for element in soup.find_all(text=True):
        clean_element_text = clean_text(element)
        if clean_element_text:
            text.append(clean_element_text)

    return ' '.join(text)

def extract_items(document, company):
    """
    Organizes the given document into a dictionary, separated by item number

    Args:
        document (String): document text
        company (String): company ticker

    Returns:
        dict: organized dictionary (Key -> Item, Value -> Content)
    """
    items_dict = {}

    doc = document.lower()

    # Cutting off the table of contents
    start_index = doc.find('item 1. business')
    
    # Exception case for MSFT filings (Earlier filings has "Item 1. Business" in table of contents)
    if company == 'MSFT':
        start_index = doc.find('item 1. business general')
    
    # Exception case for 2023 MSFT filing (text shows "Item 1. B USINESS GENERAL...")
    if start_index == -1:
        start_index = doc.find('item 1. b')

    doc = doc[start_index:]
    document = document[start_index:]

    for i in range(len(items)):
        start_index = doc.find(items[i]) + len(items[i])
        if i == len(items)-1:
            end_index = doc.find('signature title') if company == 'MSFT' else doc.find('signatures pursuant')
        else:
            end_index = doc.find(items[i+1])

        if start_index == len(items[i])-1:
            items_dict[items[i]] = 'None.'
            continue
        elif end_index == -1:
            end_index = doc.find(items[i+2]) if not i == len(items)-2 else doc.find('signatures pursuant')

        item_content = document[start_index:end_index]

        items_dict[items[i]] = item_content
        #items_dict[items[i]] = item_content[:20] + "..." + item_content[len(item_content)-20:]

    return items_dict

def deeper_clean_text(text):
    """
    Function to clean deeper text after organizing

    Args:
        text (String): text to clean

    Returns:
        String: cleaned text
    """
    text = re.sub(r'\b[\d\.\/]+\b', ' ', text)  # Remove numbers and terms like 10-Q, 10-K, etc.
    text = re.sub(r'\b\w{1,2}\b', ' ', text)  # Remove words of 1-2 letters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space

    return text

# Function to see progress through saved files. Not necessary for final product but nice to have
def save_document(content, save_path):
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(content)

companies_dict = {}

# Process each company's filings
for company in companies:
    company_path = os.path.join("sec-edgar-filings", company, "10-K")    
    companies_dict[company] = {}
    for root, dirs, files in os.walk(company_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(root)
            print(dirs)
            print(files)
            
            # Extract the portion of the full-submission.txt that contains the actual texts
            document = extract_10k_document(file_path)

            # Clean the extracted document
            document = clean_and_format_html(document)
            
            cleaned_root = "cleaned-" + root
            cleaned_path = os.path.join(cleaned_root, '10k_cleaned.txt')
            # os.makedirs(cleaned_root)
            # save_document(document, cleaned_path)
            
            # Organize the 10-K filing by Items
            items_dict = extract_items(document, company)

            for item in items:
                items_dict[item] = deeper_clean_text(items_dict[item])
            
            year = file_path[40:42] if company == 'GOOGL' else file_path[39:41]
            year = '19' + year if int(year) > 90 else '20' + year
            companies_dict[company][year] = items_dict

            # Save the whole dictionary
            # save_document(str(items_dict), os.path.join(cleaned_root, '10K_items.txt'))

#########################################################

"""
Text Analysis

companies_dict now contains all companies' 10-K Filings organized into a dictionary by items, 
we perform text analysis through different methods
"""

"""
Sentiment Analysis

Using Hugging Face's Transformers, specifically the distilroberta-finetuned-financial-news-sentiment-analysis model.
Used matplotlib.pyplot for visualization
"""

from transformers import pipeline, AutoTokenizer
import torch
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')
classifier = pipeline('sentiment-analysis', model='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')

def analyze_sentiments(text):
    """
    Perform sentiment analysis on a single text, handling long texts appropriately."

    Args:
        text (String): text to perform sentiment analysis on

    Returns:
        tuple: tuple of (sentiment, score)
    """
    # Prepare the text input for the model, managing the max token size limit
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Predict sentiment
    with torch.no_grad():  # Disable gradient calculation for inference
        results = classifier.model(**inputs)
    
    # Interpret the model's output
    sentiment = 'POSITIVE' if results.logits[0][0] > results.logits[0][1] else 'NEGATIVE'
    score = torch.nn.functional.softmax(results.logits, dim=-1).max().item()

    return sentiment, score

def analyze_company_data():
    """ 
    Analyze sentiment across multiple texts, organizing by company and year."

    Returns:
        dict: dictionary of sentiment analysis score (confidence score multiplied by 1 or -1 depending on sentiment)
    """
    results = {}

    """
    Analyzes these itmes:
        Item 1. Business
        Item 1a. Risk Factors
        Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
        Item 7a. Quantitative and Qualitative Disclosures About Market Risk
        Item 9. Changes in and Disagreements with Accountants on Accounting and Financial Disclosure
        Item 9A. Controls and Procedures
    """
    
    key_items = ['item 1. ', 'item 1a. ', 'item 7. ', 'item 7a. ', 'item 9. ', 'item 9a. ' ]

    for company, years_items in companies_dict.items():
        results[company] = {}

        for year, items in years_items.items():
            combined_confidence = []

            for item in key_items:
                item_content = items.get(item, '')
                if 'None.' not in item_content:
                    sentiment, confidence = analyze_sentiments(item_content)
                    print("Before: " + sentiment + " " + str(confidence))
                    confidence *= -1 if sentiment == 'NEGATIVE' else 1
                    print("After: " + sentiment + " " + str(confidence))
                    combined_confidence.append(confidence)

            average_confidence = sum(combined_confidence) / len(combined_confidence)
            results[company][year] = average_confidence
            
    return results

def visualize_sentiments(results):
    """
    Visualize the sentiment analysis results.


    Args:
        results (dict): dictionary containing sentiment analysis scores
    """
    for company, data in results.items():
        fig, ax = plt.subplots(figsize=(15,4))

        years = list(data.keys())
        years.sort()
        print(years)
        
        #sentiments = [1 if data[year][0] == 'POSITIVE' else -1 for year in years]
        confidences = [data[year] for year in years]
        
        plt.plot(years, confidences, label=f'{company} Sentiment', marker='o', linestyle='-', color='blue')
        plt.title(f'Yearly Sentiment Analysis for {company}')
        plt.xlabel('Year')
        plt.ylabel('Sentiment Score (Confidence Weighted)')
        plt.grid(True)
        plt.legend()
        plt.show()

results = analyze_company_data()
visualize_sentiments(results)

"""
Trend Analysis

WARNING: This is an extremely costly function and may take hours to finish

Not tested out yet for longer texts -- runtime too long
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def advanced_summarize(text):
    """
    Summarizer

    Args:
        text (String): text to summarize

    Returns:
        String: summarized text
    """
    parts = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = []
    for part in parts:
        input_ids = tokenizer(part, return_tensors="pt", truncation=True, max_length=1024)['input_ids']
        output = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return ' '.join(summaries)

def trend_analysis(item_key):
    """
    Trend Analysis

    Args:
        item_key (String): the item to analyze
    """
    for company, years_data in companies_dict.items():
        print(f"\nTrend Analysis for {company} - {item_key}")
        for year, data in years_data.items():
            if item_key in data and data[item_key]:
                try:
                    summary = advanced_summarize(data[item_key])
                    print(f"{year} Summary: {summary}")
                except Exception as e:
                    print(f"Error processing {year} data for {company}: {str(e)}")
            else:
                print(f"{year} Summary: No data available.")

# trend_analysis('item 10. ')

"""
Risk Factor Network Analysis

WARNING: This is an extremely costly function and may take up to 10 minutes to finish

Tested at threshold = 1 -- Network too messy
"""

import re
import matplotlib.pyplot as plt
import networkx as nx
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Adding custom stopwords
custom_stopwords = set(['may', 'company', 'also', 'could', 'would'])
stopwords = ENGLISH_STOP_WORDS.union(custom_stopwords)

def preprocess_text(text):
    """
    Lowercase, remove punctuation, and tokenize text.

    Args:
        text (String): text

    Returns:
        String: preprocessed text
    """
    text = re.sub(r'[^\w\s]', '', text.lower()) 
    tokens = [word for word in word_tokenize(text) if word not in stopwords]
    return tokens

def build_network(item_key='item 1a. '):
    """
    Build Network

    Args:
        item_key (str, optional): desired item. Defaults to 'item 1a. '.

    Returns:
        Graph() : Graph
    """
    G = nx.Graph()
    
    for company, years_data in companies_dict.items():
        for year, data in years_data.items():
            if item_key in data:
                text = data[item_key]
                tokens = preprocess_text(text)
                unique_tokens = set(tokens)  

                # Add nodes and edges between all unique tokens in this section
                for token1 in unique_tokens:
                    for token2 in unique_tokens:
                        if token1 != token2:
                            if G.has_edge(token1, token2):
                                G[token1][token2]['weight'] += 1
                            else:
                                G.add_edge(token1, token2, weight=1)
    
    return G

def visualize_network(G, threshold=2):
    """
    Visualize the network with an optional threshold for edge weights.

    Args:
        G (Graph()): Graph
        threshold (int, optional): threshold for the network. Higher up means stronger connections Defaults to 2.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)  

    # Draw nodes and edges with weights above threshold
    edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > threshold]
    weights = [G[u][v]['weight'] for (u, v) in edges]
    
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=50)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.title("Risk Factor Co-occurrence Network")
    plt.axis('off')
    plt.show()

# Build the network
# G = build_network(item_key='item 1a. ')

# Visualize the network with a threshold to only show stronger connections
# visualize_network(G, threshold=3)

"""
Keyword Tracking Over Time

Tracks specific keywords or phrases relevant to interests over time, which can be indicative of 
emerging trends or shifting focuses without needing to process the entire content deeply
"""

def keyword_tracking(keywords):
    """
    Keyword Tracking

    Args:
        keywords (list(str)): keywords
    """
    # Initialize results dictionary
    results = {company: {keyword: [] for keyword in keywords} for company in companies_dict}
    
    # Collect counts for each keyword by year
    for company, years_data in companies_dict.items():
        for year, data in years_data.items():
            content = " ".join(data.values()) 
            content = content.lower()  
            for keyword in keywords:
                count = content.count(keyword.lower())
                results[company][keyword].append((int(year), count)) 

    # Plotting results
    for company, keywords_data in results.items():
        plt.figure(figsize=(10, 5))
        for keyword, counts in keywords_data.items():
            counts.sort() 
            years = [year for year, count in counts]
            values = [count for year, count in counts]
            plt.plot(years, values, marker='o', label=f'{keyword}')

        plt.title(f'Keyword Frequency Over Time for {company}')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.show()

keywords = ['cybersecurity', 'data privacy', 'sustainability']
keyword_tracking(keywords)

"""
Section Length Over Time

Visualizes the length of desired section over time
"""

def visualize_section_length(item_key):
    for company, years_data in companies_dict.items():
        years = sorted(years_data.keys())
        lengths = [len(years_data[year][item_key]) if item_key in years_data[year] else 0 for year in years]

        plt.figure(figsize=(15, 5))
        plt.bar(years, lengths, color='skyblue')
        plt.title(f'Length of {item_key} over time for {company}')
        plt.xlabel('Year')
        plt.ylabel('Character Count')
        plt.show()

visualize_section_length('item 1a. ')