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
            tab1, tab2, tab3, tab4 = st.tabs(["Cross Links", "Summary", "Categories", "Configure Links"])
            
            with tab1:
                st.subheader("Cross Links Preview")
                st.dataframe(cross_links_df.head(10))
                
            with tab2:
                st.subheader("Category Summary")
                st.dataframe(summary_df.sort_values('Link Count', ascending=True).head(10))
                
            with tab3:
                st.subheader("Categories by Link Count")
                st.dataframe(categories_df.head(10))
                
            with tab4:
                st.subheader("Configure Category Cross-Links")
                
                # Get unique categories for selection
                unique_categories = summary_df[['Category', 'Title']].drop_duplicates()
                category_options = unique_categories['Title'].tolist()
                
                # Source category selection
                st.markdown("### 1. Select source category")
                source_category = st.selectbox(
                    "Source category (where the link will be placed)",
                    options=category_options
                )
                
                # Get source category details
                source_row = summary_df[summary_df['Title'] == source_category].iloc[0]
                st.info(f"Selected source: {source_category} (Current link count: {source_row['Link Count']})")
                
                # Target category selection
                st.markdown("### 2. Select target categories")
                st.info("Choose categories you want to link to from the source category")
                
                # Filter out the source from potential targets
                target_options = [c for c in category_options if c != source_category]
                
                # Show current links for this source
                current_links = cross_links_df[cross_links_df['Source Title'] == source_category]
                if not current_links.empty:
                    st.markdown("#### Current target links:")
                    current_targets = current_links['Target Title'].tolist()
                    for idx, target in enumerate(current_targets, 1):
                        st.markdown(f"{idx}. {target}")
                
                # Allow selecting multiple target categories
                selected_targets = st.multiselect(
                    "Select target categories to link to",
                    options=target_options,
                    default=[]
                )
                
                # Set link count limit
                max_selections = 10
                if len(selected_targets) > max_selections:
                    st.warning(f"You've selected {len(selected_targets)} targets. Consider limiting to {max_selections} for better SEO practices.")
                
                # Apply custom links button
                if st.button("Apply Custom Links", type="primary"):
                    # 1. Remove existing links for this source
                    mask = cross_links_df['Source Title'] != source_category
                    filtered_df = cross_links_df[mask].copy()
                    
                    # 2. Add new custom links
                    new_links = []
                    source_url = source_row['Redirect URL']
                    
                    for target in selected_targets:
                        target_row = summary_df[summary_df['Title'] == target].iloc[0]
                        target_url = target_row['Redirect URL']
                        
                        new_links.append({
                            'Target URL': target_url,
                            'Parent for Redirect URL': target_row.get('Parent Category', ''),
                            'Redirect URL': source_url,
                            'Parent Category': source_row.get('Parent Category', ''),
                            'Category': source_row['Category'],
                            'Subcategory': source_row.get('Subcategory', ''),
                            'Title': source_row['Title'],
                            'avg_monthly_searches': source_row.get('avg_monthly_searches', 0),
                            'position': 0,
                            'Seasonality': source_row.get('Seasonality', ''),
                            'Count': source_row.get('Count', 0),
                            'Link Count': len(selected_targets)
                        })
                    
                    # Create new links dataframe and append to filtered results
                    if new_links:
                        new_links_df = pd.DataFrame(new_links)
                        cross_links_df = pd.concat([filtered_df, new_links_df], ignore_index=True)
                        
                        # Update summary and categories dataframes
                        summary_df, categories_df = generate_summaries(cross_links_df)
                        
                        st.success(f"Successfully updated links for {source_category}!")
                        st.info("The changes will be reflected in the Excel report when you download it.")
            
            # Add interactive table for custom edits
            st.header("Interactive Link Editor")
            st.markdown("""
            Use this section to manually edit, add or remove specific cross-links before downloading the final report.
            This gives you full control over the cross-linking strategy.
            """)
            
            # Create dataframe editor for manual adjustments
            edited_df = st.data_editor(
                cross_links_df,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True
            )
            
            # Update button for custom edits
            if st.button("Update Cross-Links from Editor"):
                cross_links_df = edited_df.copy()
                # Regenerate summary dataframes
                summary_df, categories_df = generate_summaries(cross_links_df)
                st.success("Successfully updated cross-links!")
                
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
