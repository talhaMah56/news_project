import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
import os
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple


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


def adf_test(series: pd.Series, significance: float = 0.05) -> None:
    """Perform the Augmented Dickey-Fuller test for stationarity.

    Parameters:
        series (pd.Series): The time series to test for stationarity.
        significance (float): The significance level for the test (default: 0.05).

    Returns:
        None: Results are printed to the console.
    """
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    if result[1] <= significance:
        print("Result: The series is stationary.")
    else:
        print("Result: The series is not stationary.")


def plot_pacf_for_series(series: pd.Series,
                         lags: int = 20,
                         verbose: bool = False,
                         image_suffix: str = "",
                         use_exp: bool = False) -> None:
    """Plot the partial autocorrelation function (PACF) for a time series.

    Parameters:
        series (pd.Series): The time series to plot.
        lags (int): Number of lags to include in the PACF plot (default: 20).
        verbose (bool): If True, show the plot inline; else, save to disk.
        image_suffix (str): Suffix for the saved image filename.
        use_exp (bool): Whether exponential transformation is used.

    Returns:
        None: The plot is either displayed or saved to 'images/arima_pacf.png'.
    """
    fig1 = plot_pacf(series.to_list(), lags=lags, alpha=0.05)
    if (use_exp):
        plt.title(f'Partial Autocorrelation Function for Exponential ARIMA', fontsize=14)
    else:
        plt.title(f'Partial Autocorrelation Function for Linear ARIMA', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('Partial Autocorrelation', fontsize=12)

    if verbose:
        plt.show()
    else:
        os.makedirs('images', exist_ok=True)
        fig1.savefig(f'images/arima_pacf_{image_suffix}.png')
    plt.close()


def fit_arima_model(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    order: tuple = (1, 1, 1),
                    verbose: bool = False,
                    image_suffix: str = "",
                    use_exp: bool = False) -> tuple:
    """Fit an ARIMA model to the time series data.

    Parameters:
        train_df (pd.DataFrame): Training data with 'Frequency' column.
        test_df (pd.DataFrame): Test data with 'Frequency' column.
        order (tuple): ARIMA order (p, d, q).
        verbose (bool): If True, show plots inline; else, save to disk.
        image_suffix (str): Suffix for saved files.
        use_exp (bool): If True, apply log transformation.

    Returns:
        tuple: (fitted model, forecast DataFrame)
    """
    # Store original data for metrics calculation
    original_train = train_df['Frequency'].copy()
    original_test = test_df['Frequency'].copy()

    # Apply log transformation if requested
    if use_exp:
        train_df['Frequency'] = np.log(train_df['Frequency'])
        test_df['Frequency'] = np.log(test_df['Frequency'])

    model = ARIMA(train_df["Frequency"], order=order)
    model_fit = model.fit()
    print(f"ARIMA Model Summary:\n{model_fit.summary()}")

    # Get fitted values for training data
    fitted_values = model_fit.fittedvalues

    # Get forecast for test data
    forecast = model_fit.get_forecast(steps=len(test_df))
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Transform back if exponential was used
    if use_exp:
        fitted_values_original = np.exp(fitted_values)
        forecast_mean_original = np.exp(forecast_mean)
        forecast_ci_original = np.exp(forecast_ci)
    else:
        fitted_values_original = fitted_values
        forecast_mean_original = forecast_mean
        forecast_ci_original = forecast_ci

    # Calculate metrics for training data
    # Remove NaN values for fitted values calculation
    valid_indices = ~np.isnan(fitted_values_original)
    train_actual = original_train[valid_indices].values
    train_predictions = fitted_values_original[valid_indices].values
    train_metrics = calculate_metrics(train_actual, train_predictions)

    # Calculate metrics for test data
    test_actual = original_test.values
    test_predictions = forecast_mean_original.values
    test_metrics = calculate_metrics(test_actual, test_predictions)

    # Print results
    model_type = "Exponential (Log-transformed)" if use_exp else "Linear"
    print(f"\n=== ARIMA Model Results ({model_type}) ===")
    print(f"Order: {order}")
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
    results_file = f'results/arima_metrics_{"exp" if use_exp else "lin"}_{image_suffix}.txt'
    with open(results_file, 'w') as f:
        f.write(f"ARIMA Model Results ({model_type})\n")
        f.write(f"Order: {order}\n")
        f.write(f"Image suffix: {image_suffix}\n")
        f.write(f"\nTraining Metrics:\n")
        f.write(f"  R² Score: {train_metrics['R2']:.4f}\n")
        f.write(f"  L2 (MSE): {train_metrics['L2_MSE']:.4f}\n")
        f.write(f"  Chi²:     {train_metrics['Chi2']:.4f}\n")
        f.write(f"\nTesting Metrics:\n")
        f.write(f"  R² Score: {test_metrics['R2']:.4f}\n")
        f.write(f"  L2 (MSE): {test_metrics['L2_MSE']:.4f}\n")
        f.write(f"  Chi²:     {test_metrics['Chi2']:.4f}\n")

    # Create forecast DataFrame
    forecast_df = forecast_ci_original.copy()
    forecast_df['Predicted'] = forecast_mean_original

    # Plot
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(train_df.index, original_train, label='Train')
    plt.plot(train_df.index, fitted_values_original, label='Fitted', linestyle=':')
    plt.plot(test_df.index, original_test, label='Test')
    plt.plot(forecast_df.index, forecast_df['Predicted'], label='Forecast', linestyle='--')
    plt.fill_between(
        forecast_df.index,
        forecast_df.iloc[:, 0],  # lower bound
        forecast_df.iloc[:, 1],  # upper bound
        color='gray',
        alpha=0.3)

    # Add metrics to plot title
    title = f'ARIMA Forecast ({model_type}) vs Test Data\n'
    title += f'Test R²: {test_metrics["R2"]:.3f}, MSE: {test_metrics["L2_MSE"]:.3f}'
    plt.title(title)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()

    # Display or save the plot
    if verbose:
        plt.show()
    else:
        os.makedirs('images', exist_ok=True)
        suffix = f'{"exp" if use_exp else "lin"}_{image_suffix}'
        fig1.savefig(f'images/arima_forecast_{suffix}.png')
    plt.close()

    return model_fit, forecast_df


def run_arima_analysis(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       order: tuple = (2, 1, 2),
                       verbose: bool = False,
                       image_suffix: str = "",
                       use_exp: bool = False):
    """Main function to run ARIMA analysis using provided dataframes.

    Parameters:
        train_df (pd.DataFrame): Training data with 'Timestamp' and 'Frequency' columns.
        test_df (pd.DataFrame): Test data with 'Timestamp' and 'Frequency' columns.
        order (tuple): ARIMA order (p, d, q) (default: (2, 1, 2)).
        verbose (bool): If True, show plots inline; else, save to disk.
        image_suffix (str): Suffix for saved files.
        use_exp (bool): If True, apply log transformation.

    Returns:
        tuple: (fitted ARIMA model, forecast DataFrame)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df.set_index('Timestamp', inplace=True)

    # Create a complete monthly date range from the min to max timestamp
    full_month_range = pd.date_range(start=train_df.index.min(),
                                     end=train_df.index.max(),
                                     freq='M')

    # Reindex to the full monthly range (this will insert NaNs for missing months)
    train_df = train_df.reindex(full_month_range)

    # Interpolate missing values
    train_df = train_df.interpolate(method='time')

    # Set frequency for modeling
    train_df.index.freq = 'M'

    test_df.set_index('Timestamp', inplace=True)
    test_df.index.freq = 'M'

    print("\nADF Test Results:")
    adf_test(train_df['Frequency'])

    suffix = image_suffix
    if use_exp:
        suffix += '_exp'
    else:
        suffix += '_lin'

    plot_pacf_for_series(train_df['Frequency'],
                         verbose=verbose,
                         image_suffix=suffix,
                         use_exp=use_exp)

    model, forecast = fit_arima_model(train_df,
                                      test_df,
                                      order=order,
                                      verbose=verbose,
                                      image_suffix=suffix,
                                      use_exp=use_exp)
    return model, forecast


def compare_arima_models(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         order: tuple = (2, 1, 2),
                         image_suffix: str = "") -> Tuple[tuple, tuple]:
    """Compare linear and exponential ARIMA models."""
    print("Fitting Linear ARIMA Model...")
    model_lin, forecast_lin = run_arima_analysis(train_df, test_df,
                                                 order=order,
                                                 image_suffix=image_suffix,
                                                 use_exp=False)

    print("\nFitting Exponential ARIMA Model...")
    model_exp, forecast_exp = run_arima_analysis(train_df, test_df,
                                                 order=order,
                                                 image_suffix=image_suffix,
                                                 use_exp=True)

    return (model_lin, forecast_lin), (model_exp, forecast_exp)