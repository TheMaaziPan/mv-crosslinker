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
        # Read data with correct template
        df = pd.read_csv(uploaded_file, sep='\t') if '\t' in uploaded_file.getvalue().decode('utf-8')[:100] else pd.read_csv(uploaded_file)
        
        # Verify required columns exist
        required_columns = ['URL', 'Type', 'Category', 'Subcategory', 'Title', 
                           'Parent Category', 'Count', 'Meta Description', 'Description']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter Hybris products
        df = df[df['Type'] == 'Hybris'].copy()
        
        # Generate keywords from Title
        df['keyword'] = df['Title'].apply(
            lambda x: clean_string_and_remove_stopwords(str(x)) if pd.notna(x) else None)
        
        # Generate mock volumes (aligned with your template)
        volume_df = generate_mock_volumes(
            keywords=df['keyword'].dropna().unique(),
            months=['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
                   'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
        )
        
        # Merge with volume data
        df = pd.merge(
            df,
            volume_df[['keyword', 'avg_monthly_searches']],
            on='keyword',
            how='left'
        ).fillna(0)
        
        # Process crosslinks with your exact column names
        crosslinks_df = process_crosslinks(
            df.rename(columns={
                'URL': 'Redirect URL',
                'Title': 'Target Title'
            })
        )
        
        # Generate category summary
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
            file_name="newlook_crosslinks.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Show previews
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Cross Links")
            st.dataframe(crosslinks_df.sort_values('Similarity', ascending=False).head())
        
        with col2:
            st.subheader("Top Categories")
            st.dataframe(category_df.sort_values('Link Count', ascending=False).head())
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("""
        Your file must match this exact format:
        - URL (e.g., https://www.newlook.com/uk/women/dresses)
        - Type (must contain 'Hybris')
        - Category (e.g., Dresses)
        - Subcategory (e.g., Summer) 
        - Title (e.g., Dress)
        - Parent Category (e.g., Women)
        - Count (numeric)
        - Meta Description
        - Description
        """)
