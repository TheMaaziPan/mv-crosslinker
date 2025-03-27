import streamlit as st
import pandas as pd
from functions import (
    clean_string_and_remove_stopwords,
    generate_mock_volumes,
    process_crosslinks
)
import io

# App Config
st.set_page_config(layout="wide")
st.title("SEO Cross-Link Generator")

# File Upload
uploaded_file = st.file_uploader("Upload ALL_DATA.csv", type="csv")

if uploaded_file:
    try:
        # Read and clean data
        df = pd.read_csv(uploaded_file)
        
        # Standardize columns - ensure these match your CSV
        df = df.rename(columns={
            'URL': 'Redirect URL',
            'Product Name': 'Title',  # Adjust based on your CSV
            'Product Category': 'Category'  # Adjust based on your CSV
        })
        
        # Filter and clean
        required_cols = ['Redirect URL', 'Title', 'Category', 'Subcategory']
        df = df[df['Type'] == 'Hybris'][required_cols].dropna()
        
        # Generate keywords
        df['keyword'] = df['Title'].apply(
            lambda x: clean_string_and_remove_stopwords(x) if isinstance(x, str) else None)
        
        # Generate mock data
        volume_df = generate_mock_volumes(df['keyword'].dropna().unique())
        
        # Merge data
        df = pd.merge(
            df,
            volume_df[['keyword', 'avg_monthly_searches']],
            on='keyword',
            how='left'
        ).fillna(0)
        
        # Process crosslinks - ensure consistent column names
        crosslinks_df = process_crosslinks(df.rename(columns={
            'Title': 'Target Title',
            'Redirect URL': 'Target URL'
        }))
        
        # Generate category summary - use existing columns
        if not crosslinks_df.empty:
            category_df = (crosslinks_df
                          .groupby(['Target Title', 'Source Title'])
                          .size()
                          .reset_index(name='Link Count'))
        else:
            category_df = pd.DataFrame(columns=['Target Title', 'Source Title', 'Link Count'])
        
        # Create Excel output
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            crosslinks_df.to_excel(writer, sheet_name='Cross Links', index=False)
            df.to_excel(writer, sheet_name='Product Data', index=False)
            category_df.to_excel(writer, sheet_name='Category Links', index=False)
        
        # Download button
        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name="crosslink_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Show previews
        st.subheader("Cross Links Preview")
        st.dataframe(crosslinks_df.head())
        
        st.subheader("Category Summary")
        st.dataframe(category_df.head())
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.error("Please check your CSV columns match the expected format:")
        st.code("""
        Required columns:
        - URL (will be renamed to Redirect URL)
        - Product Name/Title
        - Category
        - Subcategory
        - Type (with 'Hybris' values)
        """)
