import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import streamlit as st
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_string_and_remove_stopwords(text):
    """
    Clean text by removing special characters, stopwords, and lemmatizing
    """
    if not isinstance(text, str):
        return None
        
    # Remove special characters and lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    
    # Tokenize and lemmatize
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    return ' '.join(words)

def get_similarity(title, description):
    """
    Calculate similarity score between title and description (0-10 scale)
    """
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([title, description])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 10  # Scale to 0-10
        return round(similarity, 2)
    except:
        return 0

def get_ad_volumes(file_path, df):
    """
    Get search volume data (mock implementation - replace with actual API call)
    """
    # Generate mock data if no file path provided
    if file_path is None:
        keywords = df['keyword'].unique()
        months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
                 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
        
        volume_data = pd.DataFrame({
            'keyword': np.repeat(keywords, len(months)),
            'geo_location': df['geo_location'].iloc[0],
            'month': months * len(keywords),
            'volume': np.random.randint(100, 5000, size=len(keywords)*len(months))
        })
        
        # Pivot to match expected format
        volume_data = volume_data.pivot(index=['keyword','geo_location'], columns='month', values='volume').reset_index()
        volume_data['avg_monthly_searches'] = volume_data.mean(axis=1)
        return volume_data
    
    # If file path provided, read from CSV
    return pd.read_csv(file_path)

def get_gsc_positions(file_path, domain):
    """
    Get Google Search Console position data (mock implementation)
    """
    if file_path is None:
        # Generate mock data
        return pd.DataFrame({
            'keyword': ['dress', 'shoes', 'jacket', 'jeans', 't-shirt'],
            'position': [2.5, 4.1, 7.8, 3.2, 5.4],
            'clicks': [1200, 850, 420, 1100, 670],
            'impressions': [15000, 12000, 8000, 14000, 9500]
        })
    
    # If file path provided, read from CSV
    return pd.read_csv(file_path)

def create_engine():
    """
    Create SQLAlchemy engine for database connection
    """
    try:
        # Get credentials from Streamlit secrets
        db_config = {
            'host': st.secrets["db_credentials"]["host"],
            'port': st.secrets["db_credentials"]["port"],
            'database': st.secrets["db_credentials"]["database"],
            'user': st.secrets["db_credentials"]["user"],
            'password': st.secrets["db_credentials"]["password"]
        }
        
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        return create_engine(connection_string)
    except:
        # Fallback to SQLite if no credentials available
        return create_engine('sqlite:///:memory:')

def extract_color_variations(title, color_codes):
    """
    Extract color variations from product title
    """
    colors = [c for c in title.lower().split() if c in color_codes]
    variations = []
    
    if colors:
        base_color = colors[0]
        for color in color_codes:
            if color != base_color:
                variation = re.sub(base_color, color, title.lower())
                variations.append(variation.title())
    
    return variations

def get_color_codes():
    """
    Fetch color codes from web
    """
    try:
        response = requests.get('http://www.nameacolor.com/COMPLETE%20COLOR%20NAMES%20TABLE.htm', timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        return pd.read_html(str(table))[0]['COLOR NAME'].str.lower().tolist()
    except:
        # Return default colors if fetch fails
        return ['red', 'blue', 'green', 'black', 'white', 'yellow', 
                'pink', 'purple', 'orange', 'brown', 'gray']
