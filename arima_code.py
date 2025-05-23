import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
import os
import matplotlib.dates as mdates


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
                         image_suffix: str = "") -> None:
    """Plot the partial autocorrelation function (PACF) for a time series.

    Parameters:
        series (pd.Series): The time series to plot.
        lags (int): Number of lags to include in the PACF plot (default: 20).
        verbose (bool): If True, show the plot inline; else, save to disk.

    Returns:
        None: The plot is either displayed or saved to 'images/arima_pacf.png'.
    """
    fig1 = plot_pacf(series.to_list(), lags=lags, alpha=0.05)
    plt.title(f'Partial Autocorrelation Function', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('Partial Autocorrelation', fontsize=12)

    if verbose:
        fig1.show()
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

    Returns:
        tuple: (fitted model, forecast DataFrame)
    """

    if use_exp:
        train_df['Frequency'] = np.log(train_df['Frequency'])
        test_df['Frequency'] = np.log(test_df['Frequency'])

    model = ARIMA(train_df["Frequency"], order=order)

    model_fit = model.fit()
    print(f"ARIMA Model Summary:\n{model_fit.summary()}")

    forecast = model_fit.get_forecast(steps=len(test_df))
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    if use_exp:
        forecast_mean = np.exp(forecast_mean)
        forecast_ci = np.exp(forecast_ci)
        test_df['Frequency'] = np.exp(test_df['Frequency'])  # To match scale

    forecast_df = forecast_ci.copy()
    forecast_df['Predicted'] = forecast_mean

    # Plot
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(
        train_df.index,
        np.exp(train_df['Frequency']) if use_exp else train_df['Frequency'],
        label='Train')
    plt.plot(
        train_df.index,
        np.exp(model_fit.fittedvalues) if use_exp else model_fit.fittedvalues,
        label='Fitted',
        linestyle=':')
    plt.plot(test_df.index, test_df['Frequency'], label='Test')
    plt.plot(forecast_df.index,
             forecast_df['Predicted'],
             label='Forecast',
             linestyle='--')
    plt.fill_between(
        forecast_df.index,
        forecast_df.iloc[:, 0],  # lower bound
        forecast_df.iloc[:, 1],  # upper bound
        color='gray',
        alpha=0.3)
    plt.legend()

    # Display or save the plot
    if verbose:
        fig1.show()
    else:
        os.makedirs('images', exist_ok=True)
        fig1.savefig(f'images/arima_forecast_{image_suffix}.png')
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
                         image_suffix=suffix)

    model, forecast = fit_arima_model(train_df,
                                      test_df,
                                      order=order,
                                      verbose=verbose,
                                      image_suffix=suffix,
                                      use_exp=use_exp)
    return model, forecast
