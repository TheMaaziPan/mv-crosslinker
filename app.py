import streamlit as st
import pandas as pd
import numpy as np
from functions import process_data_without_db
import io

# App config
st.set_page_config(layout="wide")

# File upload
uploaded_file = st.file_uploader("Upload ALL_DATA.csv", type="csv")

if uploaded_file:
    # Process data
    new_df, crosslink_df, category_df = process_data_without_db(uploaded_file)
    
    # Show preview
    st.success("Processing complete!")
    
    # Download button
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        new_df.to_excel(writer, sheet_name='Cross Links', index=False)
        crosslink_df.to_excel(writer, sheet_name='Summary', index=False)
        category_df.to_excel(writer, sheet_name='Categories', index=False)
    
    st.download_button(
        label="Download Excel Report",
        data=output.getvalue(),
        file_name="CrossLinks_Report.xlsx"
    )
