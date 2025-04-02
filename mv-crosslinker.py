import streamlit as st
import pandas as pd
import numpy as np
from functions import process_data_without_db
import io

# App config
st.set_page_config(layout="wide", page_title="SEO Cross-Link Generator")
st.title("SEO Cross-Link Generator")

# Sidebar for configuration
st.sidebar.header("Configuration")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold (0-10)",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
    help="Minimum similarity score (0-10) for categories to be linked"
)

max_links = st.sidebar.slider(
    "Maximum Links Per Category",
    min_value=1,
    max_value=20,
    value=10,
    help="Maximum number of links a category can have"
)

# File upload
st.header("Upload Data File")
st.markdown("""
Upload your data file containing category information. The file should include these columns:
- URL
- Title
- Category
- Subcategory
- Parent Category (optional)
- Count (optional)
- Link Count (optional, existing links)
""")

uploaded_file = st.file_uploader("Upload ALL_DATA.csv", type=["csv", "tsv"])

if uploaded_file:
    with st.spinner("Processing data... This may take a moment."):
        try:
            # Process data
            cross_links_df, summary_df, categories_df = process_data_without_db(
                uploaded_file,
                similarity_threshold=similarity_threshold,
                max_links=max_links
            )
            
            # Show preview
            st.success("Processing complete!")
            
            # Add metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Categories Processed", len(summary_df))
            with col2:
                st.metric("Total Cross-Links Generated", len(cross_links_df))
            with col3:
                avg_links = summary_df['Link Count'].mean()
                st.metric("Average Links Per Category", f"{avg_links:.1f}")
            
            # Show data previews
            tab1, tab2, tab3 = st.tabs(["Cross Links", "Summary", "Categories"])
            
            with tab1:
                st.subheader("Cross Links Preview")
                st.dataframe(cross_links_df.head(10))
                
            with tab2:
                st.subheader("Category Summary")
                st.dataframe(summary_df.sort_values('Link Count', ascending=True).head(10))
                
            with tab3:
                st.subheader("Categories by Link Count")
                st.dataframe(categories_df.head(10))
            
            # Download button
            st.header("Download Results")
            
            # Create Excel output
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                cross_links_df.to_excel(writer, sheet_name='Cross Links', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                categories_df.to_excel(writer, sheet_name='Categories', index=False)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name="CrossLinks_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("""
            Please make sure your file contains at least these columns:
            - URL (e.g., https://www.newlook.com/uk/women/dresses)
            - Category (e.g., Dresses)
            - Subcategory (e.g., Summer) 
            - Title (e.g., Summer Dresses)
            
            Optional columns:
            - Parent Category (e.g., Women)
            - Count (number of products)
            - Link Count (existing links)
            """)
