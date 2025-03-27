import streamlit as st
import pandas as pd
import numpy as np
from functions import (
    clean_string_and_remove_stopwords,
    get_similarity,
    get_ad_volumes,
    get_gsc_positions,
    create_engine
)
import os
from tqdm import tqdm
import io

# App Config
st.set_page_config(layout="wide")
st.title("SEO Cross-Link Generator")

# File Upload
uploaded_file = st.file_uploader("Upload ALL_DATA.csv", type="csv")

if uploaded_file:
    try:
        # Read and process data
        df_hybris = pd.read_csv(uploaded_file)
        
        # Clean data
        df_hybris = df_hybris.rename(columns={'URL': 'Redirect URL'})
        df_hybris = df_hybris.dropna(subset=['Title'])
        df_hybris = df_hybris.query("Type=='Hybris'")
        
        # Generate keywords
        df_hybris.loc[:, 'keyword'] = df_hybris.loc[:, 'Title'].apply(
            lambda x: clean_string_and_remove_stopwords(x) if isinstance(x, str) else None)
        
        # Get additional data
        volume_data = get_ad_volumes(df_hybris)
        gsc_df = get_gsc_positions()
        
        # Merge data
        crosslink_df = pd.merge(
            df_hybris,
            volume_data[['keyword', 'avg_monthly_searches']],
            on='keyword',
            how='left'
        )
        crosslink_df = pd.merge(
            crosslink_df,
            gsc_df,
            on='keyword',
            how='left'
        )
        
        # Generate crosslinks
        crosslink_cols = ['Redirect URL', 'Parent Category', 'Category', 
                         'Subcategory', 'Title', 'keyword']
        crosslink_df = crosslink_df.reindex(columns=crosslink_cols)
        
        results = []
        for i, row in tqdm(crosslink_df.iterrows(), total=len(crosslink_df)):
            current_text = ' '.join([str(row['Title']), str(row['Category']), str(row['Subcategory'])])
            
            for j, compare_row in crosslink_df.iterrows():
                if i != j:
                    compare_text = ' '.join([str(compare_row['Title']), 
                                           str(compare_row['Category']), 
                                           str(compare_row['Subcategory'])])
                    similarity = get_similarity(current_text, compare_text)
                    
                    if similarity > 5:  # Threshold
                        results.append({
                            'Target URL': row['Redirect URL'],
                            'Source URL': compare_row['Redirect URL'],
                            'Similarity': similarity,
                            'Target Title': row['Title'],
                            'Source Title': compare_row['Title']
                        })
        
        # Create output DataFrames
        new_df = pd.DataFrame(results)
        crosslink_summary = crosslink_df.copy()
        category_df = new_df.groupby(['Target Title', 'Source Title']).size().reset_index(name='Links')
        
        # Display results
        st.success("Processing complete!")
        
        # Download button
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            new_df.to_excel(writer, sheet_name='Cross Links', index=False)
            crosslink_summary.to_excel(writer, sheet_name='Summary', index=False)
            category_df.to_excel(writer, sheet_name='Categories', index=False)
        
        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name="CrossLinks_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
