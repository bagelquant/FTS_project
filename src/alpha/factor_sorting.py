"""
Fama-French 6-factor + momentum

Config:

1. lookback_period: 12 months

Process:

1. calculate z_scores for each factor
2. calculate sum_z_score for each stock
3. sort by sum_z_score
4. select top 10 stocks

Output:

1. 10 tickers each month
2. Residuals for each stock

"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.api import OLS
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def calculate_stock_returns(path: Path = Path('raw_data'),
                            output: Path = Path('output')) -> None:
    """
    calculate daily pct change price from csv file, then save to data folder
    :param path: Path to data folder
    :param output: Path to output folder
    :return: DataFrame with close price
    
    raw_data data folder structure:
    ticker1.csv  -> had columns: open, high, low, close, vol, vwap
    ticker2.csv

    read all csv files in the folder, then concat them into close
    """
    close = {}
    for f in path.glob('*.csv'):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        close[f.stem] = df['close'].copy()
    
    close_df = pd.DataFrame(close)
    # drop columns with all NaN values
    close_df = close_df.dropna(axis=1, how='all').ffill()
           
    stock_returns = close_df.pct_change().drop(close_df.index[0])
    print(stock_returns)
    stock_returns.to_csv(output / 'stock_returns.csv')


def _read_stock_returns(path: Path = Path('output/stock_returns.csv')) -> pd.DataFrame:
    """
    Read stock returns from csv file
    :param path: Path to stock_returns.csv
    :return: DataFrame with stock returns
    """
    stock_returns = pd.read_csv(path, index_col=0, parse_dates=True)
    return stock_returns


def _read_factor_returns(path: Path = Path('output/factor_returns.csv')) -> pd.DataFrame:
    """
    Read factor returns from csv file
    :param path: Path to factor_returns.csv
    :return: DataFrame with factor returns
    """
    factor_df = pd.read_csv(path, index_col=0, parse_dates=True)
    factor_df = factor_df.ffill().dropna()
    return factor_df.drop(columns=['rf'])  # Drop risk-free rate


def _single_stock_exposure_by_linear_regression(single_stock_returns: pd.Series,
                                                factor_returns: pd.DataFrame,
                                                only_include_significant_exposure: bool = True) -> tuple[pd.Series, pd.Series]:
    """
    Calculate exposure by linear regression
    :param single_stock_returns: Series with single stock returns
    :param factor_returns: DataFrame with factor returns
    :param only_include_significant_exposure: If True, only include significant exposure
    :return: tuple of Series with factor exposure and residuals
    Exposure: index: factor, value: exposure
    Residuals: index: trade_date, value: residuals
    """
    # Align the index of single stock returns and factor returns
    concat = pd.concat([single_stock_returns, factor_returns], axis=1, join='inner')
    y = concat.iloc[:, 0].copy()
    X = concat.iloc[:, 1:]
    X = sm.add_constant(X)  # Add a constant term to the predictor
    model = OLS(y, X).fit()
    exposure = model.params
    p_values = model.pvalues

    # Filter out insignificant exposures
    if only_include_significant_exposure:
        exposure = exposure[p_values < 0.05]
        # Remove constant term
        exposure = exposure[exposure.index != 'const']
    else:
        # Remove constant term
        exposure = exposure[exposure.index != 'const']
    exposure.name = single_stock_returns.name
    
    # residuals
    residuals = model.resid
    residuals.name = single_stock_returns.name
    residuals.index = y.index
    residuals = residuals.dropna()
    return exposure, residuals


def calculate_exposures_and_resid(stock_returns: pd.DataFrame,
                                  factor_returns: pd.DataFrame,
                                  only_include_significant_exposure: bool = True) -> pd.DataFrame:
    """
    Calculate exposure for each stock
    :param stock_returns: DataFrame with stock returns
    :param factor_returns: DataFrame with factor returns
    :param only_include_significant_exposure: If True, only include significant exposure
    :return: tuple of DataFrame with factor exposure and DataFrame with residuals

    Exposure: index: ticker, columns: factor, value: exposure
    Residuals: index: trade_date, columns: ticker, value: residuals
    """
    exposures = []
    resids = []
    for stock in stock_returns.columns:
        exposure, resid = _single_stock_exposure_by_linear_regression(stock_returns[stock], factor_returns,
                                                                      only_include_significant_exposure)
        exposures.append(exposure)
        resids.append(resid)
    exposures_df: pd.DataFrame = pd.DataFrame(exposures, index=stock_returns.columns).fillna(0)

    return exposures_df


def calculate_z_scores(exposure: pd.Series) -> pd.Series:
    """
    Calculate z-scores for each factor
    :param exposure: Series of factor exposure, index: ticker, value: factor exposure
    :return: Series of z-scores, index: ticker, value: z-score
    """
    mean = exposure.mean()
    std = exposure.std()
    z_scores = (exposure - mean) / std
    return z_scores


def rolling_z(stock_returns: pd.DataFrame,
              factor_returns: pd.DataFrame,
              lookback_period: int = 12,
              only_include_significant_exposure: bool = True,
              max_worker: int = 10) -> pd.DataFrame:
    """
    Stock z-scores for each month
    :param stock_returns: DataFrame with stock returns
    :param factor_returns: DataFrame with factor returns
    :param lookback_period: Lookback period in months
    :param only_include_significant_exposure: If True, only include significant exposure
    :param max_worker: Max number of workers for parallel processing
    :return: DataFrame with stock z-scores, index: month end date, columns: ticker, value: z-score
    """
    month_ends = stock_returns.resample('BME').last().index.to_list()
    factor_returns_list = [factor_returns for _ in range(len(month_ends))]
    lookback_periods = [lookback_period for _ in range(len(month_ends))]
    only_include_significant_exposures = [only_include_significant_exposure for _ in range(len(month_ends))]

    with ProcessPoolExecutor(max_workers=max_worker) as executor:
        results = list(executor.map(_single_date_z,
                                    [stock_returns] * len(month_ends),
                                    factor_returns_list,
                                    month_ends,
                                    lookback_periods,
                                    only_include_significant_exposures))

    stock_z_score = pd.concat(results, axis=1).T
    return stock_z_score


def _single_date_z(stock_returns: pd.DataFrame,
                   factor_returns: pd.DataFrame,
                   end_date: pd.Timestamp,
                   lookback_period: int = 12,
                   only_include_significant_exposure: bool = True) -> pd.Series:
    """
    Stocks z-scores for a single date
    :param stock_returns: DataFrame with stock returns
    :param factor_returns: DataFrame with factor returns
    :param end_date: End date for the lookback period
    :param lookback_period: Lookback period in months
    :param only_include_significant_exposure: If True, only include significant exposure
    :return: pd.Series, index: ticker, value: z-score

    """
    print(f"Calculating z-scores for {end_date.strftime('%Y-%m-%d')}")
    # Get the lookback period
    start_date = end_date - pd.DateOffset(months=lookback_period)

    # Get the stock returns and factor returns for the lookback period
    stock_returns_lookback = stock_returns.loc[start_date:end_date]
    factor_returns_lookback = factor_returns.loc[start_date:end_date]

    # Calculate exposures and residuals
    exposures = calculate_exposures_and_resid(stock_returns_lookback,
                                              factor_returns_lookback,
                                              only_include_significant_exposure)

    # Calculate z-scores for each factor
    z_scores = exposures.apply(calculate_z_scores, axis=0)  # type: ignore

    # Calculate sum of z-scores for each stock
    mean_z_scores = z_scores.mean(axis=1)
    mean_z_scores.name = end_date

    return mean_z_scores



def calculate_stock_z_scores(lookback_periods: list[int],
                             only_include_significant_exposure: bool = True,
                             output_path: Path = Path("output/z_scores")) -> None:
    """
    Calculate stock z-scores for each month
    :param lookback_periods: List of lookback periods in months
    :param only_include_significant_exposure: If True, only include significant exposure
    :param output_path: Path to output folder
    """
    # create output folder
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # calculate stock returns
    calculate_stock_returns()
    stock_returns = _read_stock_returns()
    factor_returns = _read_factor_returns()

    for lookback_period in lookback_periods:
        stock_z_score = rolling_z(stock_returns=stock_returns,
                                  factor_returns=factor_returns,
                                  only_include_significant_exposure=only_include_significant_exposure,
                                  lookback_period=lookback_period)
        # save csv
        stock_z_score.to_csv(output_path / f"z_scores_{lookback_period}.csv")
