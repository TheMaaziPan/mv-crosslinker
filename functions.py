import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm

# Initialize NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_string_and_remove_stopwords(text):
    """Clean product titles for keyword generation"""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = [lemmatizer.lemmatize(word) for word in text.split() 
                if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    except:
        return text.lower()

def get_similarity(text1, text2):
    """Calculate similarity score (0-10 scale)"""
    if not text1 or not text2:
        return 0
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text1, text2])
        return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 10, 2)
    except:
        return 0

def generate_mock_volumes(keywords, months=None):
    """Generate mock search volumes for products"""
    np.random.seed(42)  # Consistent results
    data = {
        'keyword': keywords,
        'avg_monthly_searches': np.random.randint(100, 5000, len(keywords))
    }
    if months:
        for month in months:
            data[month] = np.random.randint(50, 2000, len(keywords))
    return pd.DataFrame(data)

def process_crosslinks(df, similarity_threshold=5, max_links_per_category=10):
    """Generate crosslinks for products, limiting to max_links_per_category per target"""
    results = []
    
    # Track link count per category
    link_counts = {}
    
    # Process in order of low current link count and high search volume
    sorted_df = df.sort_values(
        by=['Link Count', 'avg_monthly_searches'], 
        ascending=[True, False]
    ).copy()
    
    for _, target in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
        target_url = target['Redirect URL']
        target_text = f"{target['Target Title']} {target['Category']} {target['Subcategory']}"
        
        if target_url not in link_counts:
            link_counts[target_url] = 0
            
        # Skip if already at max links
        if link_counts[target_url] >= max_links_per_category:
            continue
            
        # Find similar categories for this target, sorted by similarity
        similarities = []
        for _, source in df.iterrows():
            if target['Redirect URL'] != source['Redirect URL']:
                source_text = f"{source['Target Title']} {source['Category']} {source['Subcategory']}"
                similarity = get_similarity(target_text, source_text)
                
                if similarity >= similarity_threshold:
                    similarities.append((similarity, source))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Take only what's needed to reach max_links_per_category
        needed_links = max_links_per_category - link_counts[target_url]
        for similarity, source in similarities[:needed_links]:
            results.append({
                'Target URL': target['Redirect URL'],
                'Parent for Redirect URL': target.get('Parent Category', ''),
                'Redirect URL': source['Redirect URL'],
                'Parent Category': source.get('Parent Category', ''),
                'Category': source['Category'],
                'Subcategory': source['Subcategory'],
                'Title': source['Target Title'],
                'avg_monthly_searches': source['avg_monthly_searches'],
                'position': 0,  # Default position
                'Seasonality': source.get('Seasonality', ''),
                'Count': source.get('Count', 0),
                'Link Count': link_counts.get(source['Redirect URL'], 0)
            })
            
            # Update link counts
            link_counts[target_url] = link_counts.get(target_url, 0) + 1
    
    return pd.DataFrame(results)

def generate_summaries(crosslink_df):
    """Generate summary dataframes based on crosslinks"""
    
    # Create Summary sheet (list of all categories with their link counts)
    summary_df = crosslink_df.groupby([
        'Title', 'Redirect URL', 'Parent Category', 'Category', 'Subcategory'
    ]).agg({
        'Seasonality': 'first',
        'avg_monthly_searches': 'first',
        'position': 'first',
        'Count': 'first'
    }).reset_index()
    
    # Count links for each URL in crosslink_df
    link_counts = crosslink_df.groupby('Redirect URL').size().reset_index(name='Link Count')
    
    # Merge link counts to summary
    summary_df = pd.merge(
        summary_df,
        link_counts,
        on='Redirect URL',
        how='left'
    ).fillna({'Link Count': 0})
    
    # Create Categories sheet (list of categories with their link counts)
    categories_df = crosslink_df.groupby([
        'Parent Category', 'Category', 'Subcategory', 'Redirect URL'
    ]).size().reset_index(name='Link Count')
    
    return summary_df, categories_df

def process_data_without_db(uploaded_file, similarity_threshold=5, max_links=10):
    """
    Process uploaded data file and generate crosslinks report
    Returns three dataframes: cross_links_df, summary_df, categories_df
    """
    try:
        # Try to detect file format (CSV or TSV)
        content_sample = uploaded_file.read(1024)
        uploaded_file.seek(0)  # Reset file pointer
        
        # Check if TSV
        is_tsv = b'\t' in content_sample
        
        # Read the file with appropriate separator
        df = pd.read_csv(uploaded_file, sep='\t' if is_tsv else ',')
        
        # Verify required columns exist
        required_columns = ['URL', 'Title', 'Category', 'Subcategory']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # If 'Parent Category' is missing, create empty column
        if 'Parent Category' not in df.columns:
            df['Parent Category'] = ''
            
        # If 'Count' is missing, create empty column
        if 'Count' not in df.columns:
            df['Count'] = 0
            
        # If 'Meta Description' is missing, create empty column
        if 'Meta Description' not in df.columns:
            df['Meta Description'] = ''
            
        # If 'Description' is missing, create empty column
        if 'Description' not in df.columns:
            df['Description'] = ''
            
        # If 'Link Count' is missing, create with zeros
        if 'Link Count' not in df.columns:
            df['Link Count'] = 0
        
        # Generate keywords from Title
        df['keyword'] = df['Title'].apply(
            lambda x: clean_string_and_remove_stopwords(str(x)) if pd.notna(x) else None)
        
        # Generate search volumes
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
        ).fillna({'avg_monthly_searches': 0})
        
        # Convert numbers to numeric
        df['avg_monthly_searches'] = pd.to_numeric(df['avg_monthly_searches'], errors='coerce')
        df['Count'] = pd.to_numeric(df['Count'], errors='coerce')
        df['Link Count'] = pd.to_numeric(df['Link Count'], errors='coerce')
        
        # Add position column if missing
        if 'position' not in df.columns:
            df['position'] = 0
            
        # Add Seasonality column if missing
        if 'Seasonality' not in df.columns:
            df['Seasonality'] = ''
            
        # Process crosslinks with column mapping
        cross_links_df = process_crosslinks(
            df.rename(columns={
                'URL': 'Redirect URL',
                'Title': 'Target Title'
            }),
            similarity_threshold=similarity_threshold,
            max_links_per_category=max_links
        )
        
        # Generate summary dataframes
        summary_df, categories_df = generate_summaries(cross_links_df)
        
        return cross_links_df, summary_df, categories_df
        
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise e
