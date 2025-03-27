import re
import nltk
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import streamlit as st
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK with error handling
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError as e:
    logger.info("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLP components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Default color list as fallback
DEFAULT_COLORS = [
    'red', 'blue', 'green', 'black', 'white', 'yellow',
    'pink', 'purple', 'orange', 'brown', 'gray', 'navy',
    'teal', 'maroon', 'olive', 'silver', 'gold', 'beige'
]

def clean_string_and_remove_stopwords(text):
    """Clean and normalize text by removing stopwords and lemmatizing"""
    if not isinstance(text, str) or not text.strip():
        return None
        
    try:
        # Remove special chars and lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Tokenize, remove stopwords, and lemmatize
        words = [lemmatizer.lemmatize(word) for word in text.split() 
                if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text.lower()  # Fallback to simple lowercase

def get_similarity(title, description):
    """Calculate cosine similarity between texts (0-10 scale)"""
    if not title or not description:
        return 0.0
        
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([title, description])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 10
        return round(similarity, 2)
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        return 0.0

def get_ad_volumes(file_path, df):
    """Get search volume data with fallback to mock data"""
    try:
        if file_path:
            return pd.read_csv(file_path)
            
        # Generate mock data if no file provided
        keywords = df['keyword'].dropna().unique()
        months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
                 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
        
        data = {
            'keyword': np.repeat(keywords, len(months)),
            'geo_location': df['geo_location'].iloc[0],
            **{month: np.random.randint(100, 5000, len(keywords)) for month in months}
        }
        
        volume_data = pd.DataFrame(data)
        volume_data['avg_monthly_searches'] = volume_data[months].mean(axis=1)
        return volume_data
        
    except Exception as e:
        logger.error(f"Volume data generation failed: {e}")
        return pd.DataFrame()

def get_gsc_positions(file_path, domain):
    """Get GSC data with fallback to mock data"""
    try:
        if file_path:
            return pd.read_csv(file_path)
            
        return pd.DataFrame({
            'keyword': ['dress', 'shoes', 'jacket', 'jeans', 't-shirt'],
            'position': [2.5, 4.1, 7.8, 3.2, 5.4],
            'clicks': [1200, 850, 420, 1100, 670],
            'impressions': [15000, 12000, 8000, 14000, 9500]
        })
    except Exception as e:
        logger.error(f"GSC data generation failed: {e}")
        return pd.DataFrame()

def create_engine():
    """Create database engine with credentials from Streamlit secrets"""
    try:
        if 'db_credentials' in st.secrets:
            config = st.secrets['db_credentials']
            return create_engine(
                f"postgresql://{config['user']}:{config['password']}@"
                f"{config['host']}:{config['port']}/{config['database']}"
            )
        return create_engine('sqlite:///:memory:')
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def get_color_codes():
    """Get color names with fallback to default list"""
    try:
        response = requests.get(
            'http://www.nameacolor.com/COMPLETE%20COLOR%20NAMES%20TABLE.htm',
            timeout=5
        )
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        colors = pd.read_html(str(table))[0]['COLOR NAME'].str.lower().tolist()
        return list(set(colors + DEFAULT_COLORS))  # Combine with defaults
    except Exception as e:
        logger.warning(f"Using default color list: {e}")
        return DEFAULT_COLORS

def extract_color_variations(title, color_codes):
    """Generate color variations for a product title"""
    if not title or not color_codes:
        return []
        
    try:
        title_lower = title.lower()
        colors = [c for c in color_codes if c in title_lower]
        variations = []
        
        if colors:
            base_color = colors[0]
            for color in color_codes:
                if color != base_color and len(color.split()) == 1:  # Skip multi-word colors
                    variation = re.sub(base_color, color, title_lower)
                    variations.append(variation.title())
        
        return variations[:10]  # Limit to 10 variations
    except Exception as e:
        logger.error(f"Color variation extraction failed: {e}")
        return []
