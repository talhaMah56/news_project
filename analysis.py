"""Analysis script for exploratory data analysis (EDA) of AI-related datasets.

This script loads two datasets (MIT AI News and AI Tech Articles), performs EDA on each,
and then combines and analyzes the data together. It supports verbose output for
interactive use and can be run as a standalone script.

Functions:
    - Load and process datasets
    - Run EDA on individual datasets
    - Combine datasets and run EDA on the combined dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_load_clean import describe_df, analyze_temporal_trends, analyze_titles, analyze_sources, full_eda
from bertopic_modeling import bertopic_model, model_output
import kagglehub
from kagglehub import KaggleDatasetAdapter
import datasets
from datetime import datetime
from prophet_code import fit_prophet_model
from sklearn.model_selection import train_test_split
from arima_code import run_arima_analysis

if __name__ == "__main__":
    # Set up argument parser for optional verbose output
    parser = argparse.ArgumentParser(
        description="Run EDA with optional verbose output.")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Show output and plots")
    parser.add_argument("--no-verbose",
                        action="store_false",
                        dest="verbose",
                        help="Hide output and plots")
    args = parser.parse_args()

    # Load datasets from Kaggle and Hugging Face
    # MIT AI News dataset from Kaggle
    mit_ai_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "deepanshudalal09/mit-ai-news-published-till-2023", "articles.csv")
    # AI Tech Articles dataset from Hugging Face
    articles_ai_df = datasets.load_dataset(
        "siavava/ai-tech-articles")["train"].to_pandas()

    # Run EDA on individual datasets
    # MIT AI dataset
    mit_df = full_eda(mit_ai_df, verbose=args.verbose, dataset_name="MIT_AI")
    # AI Tech dataset
    ai_df = full_eda(articles_ai_df,
                     verbose=args.verbose,
                     dataset_name="AI_Tech")

    # Concatenate the two DataFrames vertically
    combined_df = pd.concat([mit_df, ai_df], ignore_index=True)

    # # Run EDA on the combined dataset
    full_eda(combined_df,
             verbose=args.verbose,
             dataset_name="combined_dataset")

    # # Run BERTopic modeling and get topics over time
    bertopic_model(df=combined_df, verbose=args.verbose, dataset_name="combined_dataset")

    # Generate model outputs
    topics_over_time_df = model_output(df=combined_df,
                                       verbose=args.verbose,
                                       dataset_name="combined_dataset")

    # Create a DataFrame with unique dates and their counts
    grouped = combined_df.groupby('date').size().reset_index(name='counts')
    grouped = grouped.sort_values('date')

    # Split the data
    X = grouped['date']
    y = grouped['counts']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=False)
    # Create DataFrames for training and test sets
    train_df = pd.DataFrame({'Timestamp': X_train, 'Frequency': y_train})
    test_df = pd.DataFrame({'Timestamp': X_test, 'Frequency': y_test})

    prophet_model, prophet_forecast = fit_prophet_model(
        train_df=train_df,
        test_df=test_df,
        verbose=args.verbose,
        image_suffix="articles")
    prophet_model, prophet_forecast = fit_prophet_model(
        train_df=train_df,
        test_df=test_df,
        verbose=args.verbose,
        image_suffix="articles",
        use_exp=True)

    arima_model, arima_forecast = run_arima_analysis(train_df=train_df,
                                                     test_df=test_df,
                                                     order=(2, 1, 2),
                                                     verbose=args.verbose,
                                                     image_suffix="articles")
    arima_model, arima_forecast = run_arima_analysis(train_df=train_df,
                                                     test_df=test_df,
                                                     order=(2, 1, 2),
                                                     verbose=args.verbose,
                                                     image_suffix="articles",
                                                     use_exp=True)

    topics_grouped = topics_over_time_df.groupby(
        'Timestamp').size().reset_index(name='counts')
    topics_grouped = topics_grouped.sort_values('Timestamp')

    X = topics_grouped['Timestamp']
    y = topics_grouped['counts']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=False)
    # Create DataFrames for training and test sets
    train_df = pd.DataFrame({'Timestamp': X_train, 'Frequency': y_train})
    test_df = pd.DataFrame({'Timestamp': X_test, 'Frequency': y_test})

    prophet_model, prophet_forecast = fit_prophet_model(train_df=train_df,
                                                        test_df=test_df,
                                                        verbose=args.verbose,
                                                        image_suffix="topics")
    prophet_model, prophet_forecast = fit_prophet_model(train_df=train_df,
                                                        test_df=test_df,
                                                        verbose=args.verbose,
                                                        image_suffix="topics",
                                                        use_exp=True)

    arima_model, arima_forecast = run_arima_analysis(train_df=train_df,
                                                     test_df=test_df,
                                                     order=(2, 1, 2),
                                                     verbose=args.verbose,
                                                     image_suffix="topics")
    arima_model, arima_forecast = run_arima_analysis(train_df=train_df,
                                                     test_df=test_df,
                                                     order=(2, 1, 2),
                                                     verbose=args.verbose,
                                                     image_suffix="topics",
                                                     use_exp=True)
