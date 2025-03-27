import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data_without_db(uploaded_file):
    """Process data without database dependency"""
    # Read and clean main data
    df_hybris = pd.read_csv(uploaded_file)
    df_hybris = df_hybris[df_hybris['Type'] == 'Hybris'].copy()
    
    # Generate mock data
    mock_data = {
        'Redirect URL': df_hybris['URL'].sample(5).tolist() + ['https://example.com/1'],
        'Category': ['Dresses', 'Shoes', 'Tops', 'Jeans', 'Accessories', 'Other'],
        'Subcategory': ['Summer', 'Sandals', 'T-Shirts', 'Slim', 'Bags', 'Misc'],
        'Title': ['Summer Dress', 'Comfy Sandals', 'Basic Tee', 'Skinny Jeans', 'Handbag', 'Sample'],
        'Parent Category': ['Women']*5 + ['Men'],
        'Count': [10, 5, 20, 15, 8, 3],
        'Meta Description': ['Desc']*6,
        'Description': ['Full desc']*6
    }
    
    df_accelerate = pd.DataFrame(mock_data)
    
    # Merge datasets
    df_combined = pd.concat([df_hybris, df_accelerate], ignore_index=True)
    
    # Generate crosslinks (simplified)
    crosslinks = generate_crosslinks(df_combined)
    
    return crosslinks, df_combined, pd.DataFrame()  # Return mock DataFrames

def generate_crosslinks(df):
    """Generate crosslinks based on text similarity"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df['Title'].fillna(''))
    cosine_sim = cosine_similarity(tfidf, tfidf)
    
    results = []
    for i, row in df.iterrows():
        similar_indices = cosine_sim[i].argsort()[-4:-1][::-1]
        for idx in similar_indices:
            if idx != i:
                results.append({
                    'Target URL': row['URL'],
                    'Redirect URL': df.iloc[idx]['URL'],
                    'Similarity': cosine_sim[i][idx]
                })
    
    return pd.DataFrame(results)
