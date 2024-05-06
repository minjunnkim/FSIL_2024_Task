import streamlit as st
from task1 import company_process, keyword_tracking, visualize_section_length, analyze_company_data, visualize_sentiments

st.title('10-K Filing Analysis App')

company = st.selectbox(
    "Choose a company",
    ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOGL'],
    index=0
)

analysis_type = st.selectbox(
    "Select Analysis Type",
    ['Keyword Tracking', 'Visualize Section Length', 'Sentiment Analysis (Uses Items 1, 1A, 7, 7A, 9, and 9A)'],
    index=0
)

if analysis_type == 'Keyword Tracking':
    keywords = st.text_input("Enter keywords separated by commas", "cybersecurity, data privacy, sustainability")
    keyword_list = [keyword.strip() for keyword in keywords.split(',')]

    if st.button('Analyze'):
        company_process(company)
        fig = keyword_tracking(company, keyword_list) 
        st.pyplot(fig)
elif analysis_type == 'Visualize Section Length':
    item_key = st.selectbox(
        "Select Item",
        ['Item 1.', 'Item 1A.', 'Item 1B.', 'Item 1C.', 'Item 2.', 'Item 3.', 'Item 4.', 'Item 5.', 'Item 6.', 
        'Item 7.', 'Item 7A.', 'Item 8.', 'Item 9.', 'Item 9A.', 'Item 9B.', 'Item 9C.', 'Item 10.', 'Item 11.',
        'Item 12.', 'Item 13.', 'Item 14.', 'Item 15.', 'Item 16.'],
        index=0
    )
    
    item_key = item_key.lower() + " "

    if st.button('Analyze'):
        company_process(company)
        fig = visualize_section_length(company, item_key) 
        st.pyplot(fig)
elif analysis_type == 'Sentiment Analysis':
    if st.button('Analyze'):
        company_process(company)
        results = analyze_company_data(company)
        fig = visualize_sentiments(company, results)
        st.pyplot(fig)
