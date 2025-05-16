"""Module for loading, processing, and analyzing datasets.

This module contains functions to load data from CSV URLs, combine multiple datasets,
and perform exploratory data analysis (EDA) on the datasets. The functions include:
- `load_and_process`: Load data from a CSV URL and perform initial processing.
- `combine_datasets`: Combine multiple DataFrames into a single DataFrame and save it.
- `describe_df`: Print basic DataFrame statistics and info.
- `get_english_stopwords`: Get the complete English stop words list.
- `analyze_temporal_trends`: Analyze and visualize temporal trends in the data.
- `analyze_categories`: Analyze category distribution.
- `analyze_titles`: Analyze title characteristics and content.
- `analyze_sources`: Analyze source distribution.
- `full_eda`: Run complete exploratory data analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.dates as mdates
import os
from urllib.parse import urlparse
import tldextract



def describe_df(df: pd.DataFrame, verbose: bool = False) -> None:
    """
    Print basic dataframe statistics and info
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Whether to print and show plots or just save them to disk
    """
    if verbose:
        print('=== DataFrame Basic Info ================')
        print('Shape:', df.shape)
        print('\n=== First 5 Rows =======================')
        print(df.head())
        print('\n=== Data Types =========================')
        print(df.dtypes)
        print('\n=== Missing Values =====================')
        print(df.isnull().sum())


def get_english_stopwords() -> set:
    """
    Get the complete English stop words list from CountVectorizer
    Returns:
        set: Set of English stop words
    """
    vectorizer = CountVectorizer(stop_words="english")
    return set(vectorizer.get_stop_words())

def extract_domain(url):
    # Handle URLs without protocols
    url = url.lstrip('"\'').rstrip('"\'')
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}".lower()

def analyze_temporal_trends(df: pd.DataFrame,
                            verbose: bool = False,
                            dataset_name: str = "dataset") -> pd.Series:
    """
    Analyze and visualize temporal trends in the data
    Args:
        df (pd.DataFrame): Input dataframe with date column
        verbose (bool): Whether to print and show plots or just save them to disk
        dataset_name (str): Name of the dataset
    Returns:
        pd.Series: A pandas Series containing the 'date' column from the processed data.
    """
    print('\n=== Temporal Analysis ===')
    if dataset_name != "combined_dataset":

        if dataset_name == "AI_Tech":
            date_col = 'year'
            df[date_col] = df[date_col].astype(str)
        elif dataset_name == "MIT_AI":
            date_col = 'Published Date'

        df[date_col] = pd.to_datetime(df[date_col])
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['date'] = df[date_col].dt.to_period('Y')
        else:
            # Convert year string to datetime and format as 'YYYY-MM'
            df[date_col] = pd.to_datetime(df[date_col], format='%Y')
            df['date'] = df[date_col].dt.to_period('Y')

    yearly_counts = df.groupby('date').size()
    # Ensure the index is a proper DatetimeIndex
    if not isinstance(yearly_counts.index, pd.DatetimeIndex):
        yearly_counts.index = yearly_counts.index.to_timestamp()
    print("\nMonthly article counts:")
    print(yearly_counts)


    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.family': 'sans-serif'})
    ax = plt.gca()
    ax.set_xlim([
        yearly_counts.index.min() - pd.Timedelta('365D'),
        yearly_counts.index.max() + pd.Timedelta('365D')
    ])

    plt.plot(yearly_counts.index, yearly_counts.values, marker='o')
    plt.title('AI Articles Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2)) #every 2 years 
    plt.tight_layout()

    if verbose:
        plt.show()
    else:
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        os.makedirs(images_dir, exist_ok=True)
        plt.savefig(
            os.path.join(images_dir, f'{dataset_name}_temporal_trends.png'))
    if not(pd.api.types.is_datetime64_any_dtype(df['date'])):
        df['date'] = df['date'].dt.to_timestamp(freq='Y')
    return df['date']




def analyze_titles(df: pd.DataFrame,
                   verbose: bool = False,
                   dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Analyze title characteristics and content
    Args:
        df (pd.DataFrame): Input dataframe with title column
        verbose (bool): Whether to print and show plots or just save them to disk
        dataset_name (str): Name of the dataset
    Returns:
        pd.DataFrame: A DataFrame containing title metadata including length, word count, and text.
    """
    print('\n=== Title Analysis ===')
    if dataset_name != "combined_dataset":

        # Calculate title length and word count from standardized 'title' column
        if dataset_name == "AI_Tech":
            title_col = 'title'
        elif dataset_name == "MIT_AI":
            title_col = 'Article Header'

        df['title_length'] = df[title_col].str.len()
        df['word_count'] = df[title_col].str.split().str.len()
        df['title'] = df[title_col]

        # Restore 'text' column based on dataset name
        if dataset_name == "MIT_AI":
            df['text'] = df['Article Body']

    print(
        f"\nAverage title length: {df['title_length'].mean():.1f} characters"
    )
    print(
        f"Average word count: {df['word_count'].mean():.1f} words")

    stop_words = get_english_stopwords()
    words = []
    for title in df['title']:
        words.extend(re.findall(r'\b[a-z]{3,}\b',
                                title.lower()))  # Words with 3+ letters

    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words).most_common(20)

    print("\nTop 20 title keywords (excluding stop words):")
    for word, count in word_counts:
        print(f"{word}: {count}")

    # Plot top keywords
    keywords, counts = zip(*word_counts)
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.barh(keywords[::-1], counts[::-1])  # Reverse for descending order
    plt.title('Top 20 Keywords in Titles')
    plt.xlabel('Frequency')
    plt.tight_layout()
    if verbose:
        plt.show()
    else:
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        os.makedirs(images_dir, exist_ok=True)
        plt.savefig(
            os.path.join(images_dir, f'{dataset_name}_top_keywords.png'))

    return df[['title_length', 'word_count', 'title', 'text']]


def analyze_sources(df: pd.DataFrame,
                    verbose: bool = False,
                    dataset_name: str = "dataset") -> pd.Series:
    """
    Analyze source distribution
    Args:
        df (pd.DataFrame): Input dataframe with source column
        verbose (bool): Whether to print and show plots or just save them to disk
        dataset_name (str): Name of the dataset
    """
    print('\n=== Source Analysis ===')
    if dataset_name != "combined_dataset":
        if dataset_name == "AI_Tech":
            # AI_Tech dataset does not have a 'Source' column
            print("Source analysis not available for AI_Tech dataset.")
            return pd.Series([], dtype='object')
        elif dataset_name == "MIT_AI":
            df['sources'] = df['Source'].map(lambda x: 'MIT CSAIL' if x == 'CSAIL' else x)

    top_sources = df['sources'].value_counts().head(10)
    top_sources.index = top_sources.index.str[:15] + '...'  # Truncate long source names
    print("\nTop 10 sources:")
    print(top_sources)

    plt.figure(figsize=(14, 12))
    plt.rcParams.update({'font.family': 'sans-serif'})
    top_sources.plot(kind='bar', title='Top 10 News Sources')
    plt.xlabel('Sources')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if verbose:
        plt.show()
    else:
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        os.makedirs(images_dir, exist_ok=True)
        plt.savefig(os.path.join(images_dir,
                                 f'{dataset_name}_top_sources.png'))
    return df['sources']


def analyze_urls(df: pd.DataFrame, dataset_name: str, verbose: bool = False) -> None:
    """
    Set up the 'url' column based on the dataset name and plot URL distribution.
    Args:
        verbose (bool): Whether to show plots or save them to disk.
    """
    if dataset_name == "MIT_AI":
        df['url'] = df['Url']

    # Extract domain from URLs
    df["domain"] = df["url"].apply(lambda x: extract_domain(x))
    top_domains = df['domain'].value_counts().head(10)

    print("\nTop 10 sources:")
    print(top_domains)
    
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.family': 'sans-serif'})
    top_domains.plot(kind='bar', title='Top 10 Domains')
    plt.xlabel('Domain')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if verbose:
        plt.show()
    else:
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        os.makedirs(images_dir, exist_ok=True)
        plt.savefig(os.path.join(images_dir, f'{dataset_name}_top_urls.png'))

    return df["url"]


def full_eda(df: pd.DataFrame,
             verbose: bool = False,
             dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Run complete exploratory data analysis and return a processed DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe containing raw data.
        verbose (bool): Whether to print and show plots or just save them to disk.
        dataset_name (str): Name of the dataset (e.g., "MIT_AI", "AI_Tech", "combined_dataset").

    Returns:
        pd.DataFrame: A processed DataFrame with the following columns:
            - date (datetime64[ns]): Date of the article.
            - temporal (int or float): Temporal trend value.
            - title (object): Title of the article.
            - title_length (int): Length of the title in characters.
            - word_count (int): Number of words in the title.
            - text (object): Full text of the article.
            - url (object): URL of the article.
            - sources (object): Source of the article.

    """
    print('\n' + '=' * 50)
    print(f"=== {dataset_name} EDA ===")
    print('=' * 50 + '\n')

    describe_df(df, verbose=verbose)
    date_series = analyze_temporal_trends(df, verbose=verbose, dataset_name=dataset_name)
    titles_df = analyze_titles(df, verbose=verbose, dataset_name=dataset_name)
    sources_series = analyze_sources(df, verbose=verbose, dataset_name=dataset_name)
    url_series = analyze_urls(df, verbose=verbose, dataset_name=dataset_name)

    processed_df = pd.DataFrame({
        'date': date_series,
        'title': titles_df['title'],
        'title_length': titles_df['title_length'],
        'word_count': titles_df['word_count'],
        'text': titles_df['text'],
        'url': url_series,
        'sources': sources_series
    })

    return processed_df
