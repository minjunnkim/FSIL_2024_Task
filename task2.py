import streamlit as st
from task1 import download_10k, keyword_tracking, visualize_section_length, analyze_company_data, visualize_sentiments

def main():
    st.title('10-K Filing Analysis App')

    company = st.selectbox(
        "Choose a company",
        ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOGL'],
        index=0
    )

    analysis_type = st.selectbox(
        "Select Analysis Type",
        ['Keyword Tracking', 'Visualize Section Length', 'Sentiment Analysis'],
        index=0
    )

    if st.button('Analyze'):
        if analysis_type == 'Keyword Tracking':
            keywords = st.text_input("Enter keywords separated by commas", "cybersecurity, data privacy, sustainability")
            keyword_list = [keyword.strip() for keyword in keywords.split(',')]
            fig = keyword_tracking(company, keyword_list) 
            st.pyplot(fig)
        elif analysis_type == 'Visualize Section Length':
            item_key = st.text_input("Enter the item key", "item 1a. ")
            fig = visualize_section_length(company, item_key) 
            st.pyplot(fig)
        elif analysis_type == 'Sentiment Analysis':
            results = analyze_company_data(company)
            fig = visualize_sentiments(company, results)
            st.pyplot(fig)

if __name__ == "__main__":
    main()