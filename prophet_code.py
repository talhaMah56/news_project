import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from typing import Tuple
import numpy as np

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate R², L2 (MSE), and Chi² metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary containing the calculated metrics
    """
    # R² (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)

    # L2 (Mean Squared Error)
    mse = mean_squared_error(y_true, y_pred)

    # Chi² (Chi-squared statistic)
    # Using the formula: sum((observed - expected)² / expected)
    # Adding small epsilon to avoid division by zero
    epsilon = 1e-10
    chi2 = np.sum((y_true - y_pred) ** 2 / (np.abs(y_pred) + epsilon))

    return {
        'R2': r2,
        'L2_MSE': mse,
        'Chi2': chi2
    }


def fit_prophet_model(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      verbose: bool = False,
                      image_suffix: str = "",
                      use_exp: bool = False) -> Tuple[Prophet, pd.DataFrame]:
    """Fit a Prophet model to the time series data and plot results.

    Args:
        train_df: DataFrame with 'Timestamp' and 'Frequency' columns for training.
        test_df: DataFrame with 'Timestamp' and 'Frequency' columns for testing.
        verbose: If True, show plots; otherwise, save them to files.
        image_suffix: Optional suffix for custom image filenames.
        use_exp: If True, use logistic growth model instead of linear.

    Returns:
        A tuple containing the fitted Prophet model and forecast DataFrame.
    """
    # Rename columns to match Prophet's expected format
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = train_df.rename(columns={'Timestamp': 'ds', 'Frequency': 'y'})
    test_df = test_df.rename(columns={'Timestamp': 'ds', 'Frequency': 'y'})

    # Set capacity for logistic growth
    if use_exp:
        train_df['cap'] = 10 ** 3

    # Initialize and fit the Prophet model
    model = Prophet(growth='logistic' if use_exp else 'linear')
    model.fit(train_df)

    # Generate future dataframe based on test data
    future = model.make_future_dataframe(periods=len(test_df), freq='M')
    if use_exp:
        future['cap'] = 10 ** 3

    forecast = model.predict(future)

    # Calculate metrics for training data
    train_indices = range(len(train_df))
    train_predictions = forecast.iloc[train_indices]['yhat'].values
    train_actual = train_df['y'].values
    train_metrics = calculate_metrics(train_actual, train_predictions)

    # Calculate metrics for test data
    test_indices = range(len(train_df), len(forecast))
    test_predictions = forecast.iloc[test_indices]['yhat'].values
    test_actual = test_df['y'].values
    test_metrics = calculate_metrics(test_actual, test_predictions)

    # Print results
    model_type = "Exponential (Logistic)" if use_exp else "Linear"
    print(f"\n=== Prophet Model Results ({model_type}) ===")
    print(f"Image suffix: {image_suffix}")
    print("\nTraining Metrics:")
    print(f"  R² Score: {train_metrics['R2']:.4f}")
    print(f"  L2 (MSE): {train_metrics['L2_MSE']:.4f}")
    print(f"  Chi²:     {train_metrics['Chi2']:.4f}")

    print("\nTesting Metrics:")
    print(f"  R² Score: {test_metrics['R2']:.4f}")
    print(f"  L2 (MSE): {test_metrics['L2_MSE']:.4f}")
    print(f"  Chi²:     {test_metrics['Chi2']:.4f}")
    print("=" * 50)

    # Save metrics to file
    os.makedirs('results', exist_ok=True)
    results_file = f'results/metrics_{"exp" if use_exp else "lin"}_{image_suffix}.txt'
    with open(results_file, 'w') as f:
        f.write(f"Prophet Model Results ({model_type})\n")
        f.write(f"Image suffix: {image_suffix}\n")
        f.write(f"\nTraining Metrics:\n")
        f.write(f"  R² Score: {train_metrics['R2']:.4f}\n")
        f.write(f"  L2 (MSE): {train_metrics['L2_MSE']:.4f}\n")
        f.write(f"  Chi²:     {train_metrics['Chi2']:.4f}\n")
        f.write(f"\nTesting Metrics:\n")
        f.write(f"  R² Score: {test_metrics['R2']:.4f}\n")
        f.write(f"  L2 (MSE): {test_metrics['L2_MSE']:.4f}\n")
        f.write(f"  Chi²:     {test_metrics['Chi2']:.4f}\n")

    # Plot results
    fig1 = plt.figure(figsize=(12, 6))

    plt.plot(train_df['ds'], train_df['y'], label='Train', color='green')
    plt.plot(test_df['ds'], test_df['y'], label='Test', color='blue')
    plt.plot(forecast['ds'],
             forecast['yhat'],
             label='Forecast',
             color='orange',
             linestyle='--')
    plt.fill_between(forecast['ds'],
                     forecast['yhat_lower'],
                     forecast['yhat_upper'],
                     color='gray',
                     alpha=0.3)
    plt.legend()

    # Add metrics to plot title
    plt.title(f"Prophet Forecast ({model_type}) vs Test Data\n"
              f"Test R²: {test_metrics['R2']:.3f}, MSE: {test_metrics['L2_MSE']:.3f}")
    plt.xlabel("Date")
    plt.ylabel("Value")

    fig2 = model.plot_components(forecast)

    # Show or save plots
    if verbose:
        fig1.show()
        fig2.show()
    else:
        os.makedirs('images', exist_ok=True)
        fig1.savefig(
            f'images/prophet_forecast_{"exp" if use_exp else "lin"}_{image_suffix}.png'
        )
        fig2.savefig(
            f'images/prophet_components_{"exp" if use_exp else "lin"}_{image_suffix}.png'
        )

    plt.close()
    return model, forecast


# Example usage function
def compare_models(train_df: pd.DataFrame, test_df: pd.DataFrame, image_suffix: str = ""):
    """Compare linear and exponential Prophet models."""
    print("Fitting Linear Model...")
    model_lin, forecast_lin = fit_prophet_model(train_df, test_df,
                                                image_suffix=image_suffix,
                                                use_exp=False)

    print("\nFitting Exponential Model...")
    model_exp, forecast_exp = fit_prophet_model(train_df, test_df,
                                                image_suffix=image_suffix,
                                                use_exp=True)

    return (model_lin, forecast_lin), (model_exp, forecast_exp)
