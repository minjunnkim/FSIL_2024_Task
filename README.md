# FSIL_2024_Task

The application can be accessed online at,

[text](https://fsil2024task.streamlit.app/)

## Why Streamlit?

I used streamlit because I had just used this in another project, and found that it is easy to display matplotlib figures in it. It is the simplest I thought of.

- streamlit

## Why the libraries?

For ML and NLP, I used a wide variety of libraries. Some were used in the functions shown in the website and the others are commented out in task1.py

- torch
- transformers
- scikit-learn
- nltk

I used some other commonly-used libraries to graph, and simplify calculations.

- matplotlib
- networkx
- numpy

For processing XML/HTML formatted 10-K Filing .txt files, I used below libraries to clean the data.

- lxml
- beautifulsoup4

Lastly, I chose to use the given example for downloading 10-K Filings, as it was the given one and I have no previous experience in handling 10-K Filings.

- sec_edgar_downloader
