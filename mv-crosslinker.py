import streamlit as st
import os
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from functions import *
import io
from datetime import datetime
from tqdm import tqdm

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Session state initialization
if 'processed' not in st.session_state:
    st.session_state.processed = False

# App header
st.title("SEO Cross-Link Generator")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("Client Configuration")
    client_name = st.text_input("Client Name", "New Look")
    gsc_domain = st.text_input("GSC Domain", "sc-domain:newlook.com")
    base_url = st.text_input("Base URL", "https://www.newlook.com/uk/")
    geo_location = st.text_input("Geo Location", "GB")
    max_links = st.number_input("Max Links", 15, 50, 15)
    threshold = st.slider("Seasonality Threshold", 0.1, 0.5, 0.3)
    uploaded_file = st.file_uploader("Upload ALL_DATA.csv", type="csv")

def process_data(uploaded_file, client_config):
    # Get month names
    current_month = datetime.now().month
    this_month_name = datetime(2023, current_month, 1).strftime('%B').upper()
    next_month = current_month % 12 + 1
    next_month_name = datetime(2023, next_month, 1).strftime('%B').upper()

    # Process Hybris data
    df_hybris = pd.read_csv(uploaded_file, low_memory=False).rename(columns={'URL': 'Redirect URL'}).dropna(subset=['Title']).query("Type=='Hybris'")
    df_hybris = df_hybris.reindex(columns=['Redirect URL', 'Category', 'Subcategory', 'Title', 'Parent Category', 'Count', 'Meta Description', 'Description'])
    df_hybris.drop_duplicates(subset=['Redirect URL', 'Category', 'Subcategory', 'Title', 'Parent Category', 'Meta Description', 'Description'], 
                             inplace=True, ignore_index=True)
    df_hybris.sort_values(by=['Parent Category', 'Category', 'Subcategory', 'Title'], inplace=True, ignore_index=True)
    df_hybris.loc[:, 'geo_location'] = client_config.get('geo_location')
    df_hybris.loc[:, 'keyword'] = df_hybris.loc[:, 'Title'].apply(
        lambda x: clean_string_and_remove_stopwords(x) if isinstance(x, str) else None)
    df_hybris.loc[:, 'Category'] = df_hybris.loc[:, 'Category'].apply(
        lambda x: x.replace('-', ' ').title() if isinstance(x, str) else None)

    # Process Accelerate data
    engine = create_engine()
    df_accelerate = pd.read_sql(f"""
        SELECT distinct url, category, subcategory, title, parent_category, count, meta_description, description
        FROM tool_db.cmt_view
        WHERE client = '{client_config.get('client_name')}'
        AND date = (SELECT max(date) FROM tool_db.cmt_view WHERE client = '{client_config.get('client_name')}')
        ORDER BY parent_category, category, subcategory, title""".strip(), engine).dropna(subset=['title'])
    
    df_accelerate.dropna(subset=['category', 'subcategory', 'parent_category'], how='all', inplace=True)
    df_accelerate.rename(columns={col: col.replace('_', ' ').title() for col in df_accelerate.columns}, inplace=True)
    df_accelerate.rename(columns={'Url': 'Redirect URL'}, inplace=True)

    # Process parent categories
    acc_parent = pd.read_sql(f"""
        SELECT DISTINCT 'https://www.newlook.com/uk/b'+url as url, parent_page 
        FROM accelerate_db.accelerate_created_category 
        WHERE client_name = '{client_config.get('client_name')}'
        ORDER BY url""", engine)
    
    acc_parent.loc[:, 'parent_page'] = acc_parent.loc[:, 'parent_page'].apply(
        lambda x: acc_parent.set_index('url').parent_page.to_dict().get(x.split('#')[0]) 
        if x.split('#')[0] in acc_parent.url.values else x)
    acc_parent.loc[:, 'parent_page'] = acc_parent.loc[:, 'parent_page'].apply(
        lambda x: x.split('.com/uk/')[-1])
    acc_parent.loc[:, 'parent_page'] = acc_parent.loc[:, 'parent_page'].apply(
        lambda x: x.split('.com/row/')[-1])
    acc_parent.loc[:, 'Category'] = acc_parent.loc[:, 'parent_page'].apply(
        lambda x: x.split('/')[0].replace('-', ' ').title() if isinstance(x.split('/')[0], str) else None)
    acc_parent.loc[:, 'Subcategory'] = acc_parent.loc[:, 'parent_page'].apply(
        lambda x: x.split('/')[1].title() if len(x.split('/'))>1 else None)
    acc_parent.sort_values(by=['Category', 'Subcategory'], inplace=True)

    # Process volume data
    volume_data = get_ad_volumes(None, df_hybris)  # Modified to skip file output
    volume_data.loc[:, 'avg_monthly_searches'] = volume_data.loc[:, next_month_name].fillna(0)
    
    spring_summer = ['APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER']
    autumn_winter = ['OCTOBER', 'NOVEMBER', 'DECEMBER', 'JANUARY', 'FEBRUARY', 'MARCH']
    
    volume_data.loc[:, 'seasonality_score'] = (volume_data.loc[:, spring_summer].sum(axis=1) / 
                                            volume_data.loc[:, autumn_winter].sum(axis=1))
    volume_data.loc[:, 'seasonality_score'] = volume_data.loc[:, 'seasonality_score'].replace(
        [np.nan, np.inf, -np.inf, pd.NA], 0).round(2)
    volume_data.drop(columns=spring_summer + autumn_winter + ['low_top_of_page_bid', 'competition_index', 'average_cpc'], 
                    inplace=True)
    
    volume_data.loc[:, 'Seasonality'] = pd.cut(
        volume_data['seasonality_score'],
        bins=[min(volume_data['seasonality_score']),
              (1 - client_config.get('threshold')),
              (1 + client_config.get('threshold')),
              max(volume_data['seasonality_score'])],
        labels=['A/W', 'All', 'S/S'])
    
    volume_data.loc[:, 'Seasonality'] = volume_data.loc[:, 'Seasonality'].astype(str).replace('nan', 'NA')

    # Process GSC data
    gsc_df = get_gsc_positions(None, client_config.get('gsc_domain'))  # Modified to skip file output

    # Merge data
    crosslink_df = pd.merge(
        df_hybris, 
        volume_data.loc[:, ['keyword', 'geo_location', 'Seasonality', 'avg_monthly_searches']].drop_duplicates(),
        on=['keyword', 'geo_location'], 
        how='left'
    )
    
    crosslink_df = pd.merge(crosslink_df, gsc_df, on=['keyword'], how='left')
    crosslink_df.loc[:, ['position', 'avg_monthly_searches']] = crosslink_df.loc[:, ['position', 'avg_monthly_searches']].fillna(0)
    crosslink_df.loc[:, "Allowed"] = 10
    crosslink_df.loc[crosslink_df['Count'] == 0, 'Allowed'] = 0

    # Color processing
    colour_codes = pd.read_html('http://www.nameacolor.com/COMPLETE%20COLOR%20NAMES%20TABLE.htm', 
                              header=[0])[1].loc[:, 'COLOR NAME'].str.lower().tolist()

    # Crosslink generation
    crosslink_cols = ['geo_location', 'Redirect URL', 'Parent Category', 'Category', 'Subcategory', 'Title', 
                    'Meta Description', 'Description', 'Seasonality', 'Count', 'avg_monthly_searches', 
                    'position', 'Allowed', 'keyword']
    crosslink_df = crosslink_df.reindex(columns=crosslink_cols)

    new_df = pd.DataFrame()
    for i, row in tqdm(crosslink_df.replace('', None).iterrows(), desc='Searching for Crosslinks', 
                      total=len(crosslink_df)):
        links = pd.DataFrame()
        sub_df = crosslink_df.query(f'`Redirect URL` != "{row.get("Redirect URL")}"')
        broad_match_cols = ['Redirect URL', 'Parent Category', 'Category', 'Subcategory', 'Title', 
                          'Meta Description', 'Description']
        
        if isinstance(row.get('keyword'), str):
            sub_df = sub_df.loc[sub_df.loc[:, broad_match_cols].apply(
                lambda x: x.str.contains(row.get("keyword").replace(' ', '|'), case=False, regex=True).any(), 
                axis=1)]
        else:
            sub_df = sub_df.loc[sub_df.loc[:, broad_match_cols].apply(
                lambda x: x.str.contains(row.get("Title").replace(' ', '|'), case=False, regex=True).any(), 
                axis=1)]
        
        links = pd.concat([links, sub_df], ignore_index=True)

        if not links.empty:
            for j, link_row in links.fillna('').iterrows():
                mainstring = '\n\n'.join([link_row.get('Title'), link_row.get('Meta Description'), 
                                        link_row.get('Description')])
                if isinstance(row.get('keyword'), str):
                    links.loc[j, 'similarity_score'] = get_similarity(title=row.get("Title"), 
                                                                      description=mainstring)
                else:
                    links.loc[j, 'similarity_score'] = get_similarity(title=row.get("keyword"), 
                                                                      description=mainstring)
            
            # Filter and sort links
            if not links.query('similarity_score>7.5').empty:
                links = links.query('similarity_score>7.5').sort_values(
                    by=['similarity_score', 'Allowed'], ascending=False, ignore_index=True)
            elif not links.query('similarity_score>5').empty:
                links = links.query('similarity_score>5').sort_values(
                    by=['similarity_score', 'Allowed'], ascending=False, ignore_index=True)
            else:
                links = links.sort_values(by=['similarity_score', 'Allowed'], ascending=False, ignore_index=True)
            
            links = links.head(client_config.get('MAX_LINKS') - 3)
            
            if not links.empty:
                crosslink_df.loc[crosslink_df['Redirect URL'].isin(links.loc[:, "Redirect URL"].tolist()), 
                               "Allowed"] -= 1
                links.loc[:, 'Target URL'] = row.get('Redirect URL')
                links.loc[:, 'Link Count'] = len(links)
                new_df = pd.concat([new_df, links], ignore_index=True)

    # Final processing
    crosslink_df.loc[:, 'Link Count'] = crosslink_df.loc[:, 'Title'].map(
        new_df.groupby(['Title']).size().to_dict()).fillna(0)
    
    new_df.loc[:, 'Parent for Redirect URL'] = new_df.loc[:, 'Redirect URL'].map(
        crosslink_df.set_index('Redirect URL').loc[:, 'Category'].to_dict())
    
    new_df = new_df.reindex(columns=[
        'Target URL', 'Parent for Redirect URL', 'Redirect URL', 'Parent Category', 
        'Category', 'Subcategory', 'Title', 'avg_monthly_searches', 'position', 
        'Seasonality', 'Count', 'Link Count'
    ])
    
    crosslink_df.drop(columns=['geo_location', 'Meta Description', 'Description', 'Allowed', 'keyword'], 
                    inplace=True)

    category_df = new_df.groupby(['Parent Category', 'Category', 'Subcategory', 'Redirect URL']).size().reset_index(name='Link Count')
    category_df.loc[:, 'Redirect URL'] = category_df.loc[:, 'Redirect URL'].apply(lambda x: x.split('.com/')[-1])

    return new_df, crosslink_df, category_df

# When user clicks process button
if uploaded_file and st.button("Generate Cross-Links"):
    client_config = {
        'client_name': client_name,
        'gsc_domain': gsc_domain,
        'client_base_url': base_url,
        'geo_location': geo_location,
        'MAX_LINKS': max_links,
        'threshold': threshold
    }
    
    with st.spinner("Processing data..."):
        new_df, crosslink_df, category_df = process_data(uploaded_file, client_config)
        st.session_state.processed = True
        st.session_state.new_df = new_df
        st.session_state.crosslink_df = crosslink_df
        st.session_state.category_df = category_df

# Show results if processed
if st.session_state.processed:
    st.success("Processing complete!")
    
    # Show preview tabs
    tab1, tab2, tab3 = st.tabs(["Cross Links", "Summary", "Categories"])
    
    with tab1:
        st.dataframe(st.session_state.new_df.head())
    
    with tab2:
        st.dataframe(st.session_state.crosslink_df.head())
    
    with tab3:
        st.dataframe(st.session_state.category_df.head())
    
    # Create download button
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.new_df.to_excel(writer, sheet_name='Cross Links', index=False)
        st.session_state.crosslink_df.to_excel(writer, sheet_name='Summary', index=False)
        st.session_state.category_df.to_excel(writer, sheet_name='Categories', index=False)
    
    st.download_button(
        label="Download Excel Report",
        data=output.getvalue(),
        file_name=f"{client_config['client_name']}_CrossLinks.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Add help section
with st.expander("How to use this app"):
    st.markdown("""
    1. Upload your ALL_DATA.csv file
    2. Configure client settings in the sidebar
    3. Click 'Generate Cross-Links'
    4. Preview results and download the report
    """)

st.markdown("---")
