"""Module for performing BERTopic modeling and generating visualizations.

This module contains functions to perform BERTopic modeling on a given DataFrame
and to generate and save visualizations from the BERTopic model. The module uses
Sentence Transformers for embeddings, UMAP for dimensionality reduction, and HDBSCAN
for clustering. It also supports using cuML for GPU acceleration if available.

Functions:
- `bertopic_model`: Perform BERTopic modeling on the given DataFrame.
- `model_output`: Generate and save visualizations from the BERTopic model.
"""

import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from os import environ, getcwd
import pandas as pd
import os


def bertopic_model(df: pd.DataFrame,
                   verbose: bool = False,
                   sentence_model_name: str = "all-MiniLM-L6-v2",
                   dataset_name: str = "dataset",
                   use_cuml: bool = False) -> None:
    """Perform BERTopic modeling on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the dataset.
    verbose (bool): If True, enable verbose output.
    sentence_model_name (str): The name of the sentence model to use.
    dataset_name (str): The name of the dataset.
    random_number (int): A random number for model and image filenames.
    use_cuml (bool): If True, use cuML for GPU acceleration.

    Returns:
    None
    """
    environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    environ["TOKENIZERS_PARALLELISM"] = "true"

    article_titles = df['title'].to_list()

    sentence_model = SentenceTransformer(f"{sentence_model_name}")
    embeddings = sentence_model.encode(article_titles,
                                       show_progress_bar=verbose)

    if (torch.cuda.is_available() and use_cuml):
        from cuml.manifold import UMAP
        from cuml.cluster import HDBSCAN
        umap_model = UMAP(n_components=5,
                          n_neighbors=15,
                          min_dist=0.0,
                          random_state=1335)
        hdbscan_model = HDBSCAN(min_samples=10,
                                gen_min_span_tree=True,
                                prediction_data=True)
        hdbscan_model.random_state = 1335
    else:
        from umap import UMAP
        from hdbscan import HDBSCAN
        umap_model = UMAP(n_components=5,
                          n_neighbors=15,
                          min_dist=0.0,
                          random_state=1335)
        hdbscan_model = HDBSCAN(min_samples=10,
                                gen_min_span_tree=True,
                                prediction_data=True)
        hdbscan_model.random_state = 1335

    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer()
    representation_model = KeyBERTInspired(random_state=1335)

    topic_model = BERTopic(verbose=verbose,
                           embedding_model=sentence_model,
                           umap_model=umap_model,
                           hdbscan_model=hdbscan_model,
                           vectorizer_model=vectorizer_model,
                           ctfidf_model=ctfidf_model,
                           representation_model=representation_model)

    topics, probs = topic_model.fit_transform(article_titles, embeddings)

    model_path = os.path.join(os.path.dirname(__file__), 'models')

    model_filename = f"{dataset_name}_bertopic_model"
    print('''WWWWWWWW''')
    topic_model.save(
        f"{model_path}/{model_filename}",
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=f"sentence-transformers/{sentence_model_name}")
    

def model_output(df: pd.DataFrame,
                 verbose: bool = False,
                 dataset_name: str = "dataset") -> None:
    """Generate and save visualizations from the BERTopic model.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the dataset.
    verbose (bool): If True, enable verbose output.
    dataset_name (str): The name of the dataset.
    random_number (int): A random number for model and image filenames.

    Returns:
    None
    """
    model_path = os.path.join(os.path.dirname(__file__), 'models')
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    model_filename = f"{dataset_name}_bertopic_model"
    topic_model = BERTopic.load(f"{model_path}/{model_filename}")
    if verbose:
        print(topic_model.get_topic_info())
    fig = topic_model.visualize_barchart()
    if verbose:
        fig.show()
    fig.write_image(os.path.join(images_dir, f"{model_filename}_barchart.png"))
    fig = topic_model.visualize_hierarchy(top_n_topics=50, orientation="left")
    if verbose:
        fig.show()
    fig.write_image(os.path.join(images_dir,
                                 f"{model_filename}_hierarchy.png"))
    topics_over_time = topic_model.topics_over_time(df["title"].to_list(), df["date"].to_list())
    fig = topic_model.visualize_topics_over_time(
        topics_over_time=topics_over_time, top_n_topics=8)
    if verbose:
        fig.show()
    fig.write_image(
        os.path.join(images_dir, f"{model_filename}_topics_over_time.png"))
    
    return topics_over_time
