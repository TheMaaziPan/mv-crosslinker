# SEO Cross-Linker Tool

This Streamlit application generates intelligent cross-links between category pages based on:
- Content similarity
- Search demand (monthly searches)
- Link distribution (prioritizing categories with fewer links)
- Product count

## Features

- **Smart Cross-Linking Algorithm**: Analyzes category content to identify related pages
- **Link Balancing**: Ensures no category has more than the configured maximum of links (default: 10)
- **Search Demand Prioritization**: Considers search volume in link recommendations
- **Customizable Parameters**: Adjust similarity threshold and maximum links per category
- **Excel Report Generation**: Creates a comprehensive Excel report with three sheets:
  - Cross Links: Detailed list of all recommended cross-links
  - Summary: Overview of all categories with link counts
  - Categories: Categorized view of link distribution

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```
   python nltk_setup.py
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Upload your data file (CSV or TSV) containing category information
   - Required columns: URL, Title, Category, Subcategory
   - Optional columns: Parent Category, Count, Link Count, Meta Description, Description

3. Adjust parameters in the sidebar:
   - Similarity Threshold: Minimum similarity score (0-10) for categories to be linked
   - Maximum Links Per Category: Limit on links per category

4. Review the generated cross-links and download the Excel report

## Data Format

Your input file should contain at least the following columns:
- **URL**: The page URL (e.g., https://www.example.com/women/dresses)
- **Title**: The page title (e.g., "Summer Dresses")
- **Category**: Main category (e.g., "Dresses")
- **Subcategory**: Secondary category (e.g., "Summer")

Optional but recommended columns:
- **Parent Category**: Top-level category (e.g., "Women")
- **Count**: Number of products in that category
- **Link Count**: Number of existing links (if any)
- **Meta Description**: Page meta description
- **Description**: Full page description

## How It Works

1. **Content Analysis**: The application analyzes category titles, descriptions, and category hierarchies
2. **Similarity Calculation**: Uses TF-IDF and cosine similarity to find related categories
3. **Link Prioritization**: Prioritizes categories with few links and high search demand
4. **Report Generation**: Creates a comprehensive Excel report with all necessary data

## Troubleshooting

If you encounter an error like "Missing required columns," make sure your CSV has the required columns (URL, Title, Category, Subcategory).

For large files, the processing may take several minutes. Please be patient while the application processes your data.
