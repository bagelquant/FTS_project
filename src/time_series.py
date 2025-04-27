"""
Time series model module


1. Each month, pick top n z-scores stock
2. Find the ARIMA model p, q for each stock
"""

import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from concurrent.futures import ProcessPoolExecutor



with open("output/significant_alphas_list.txt") as f:
    # read the file, each line is an alpha name
    OVERALL_SIGNIFICANT_ALPHAS = f.read().splitlines()


def sort_top_n_z_score(z_scores_df: pd.DataFrame,
                       n: int = 10) -> pd.DataFrame:
    """
    Get top n z-scores ticker
    :param z_scores_df: DataFrame with z-scores, index date, columns tickers
    :param n: number of top z-scores to return
    :return: DataFrame with top n z-scores, index date, columns 1-n, values tickers
    """
    # for each row, sort the z-scores and get the top n
    top_n = pd.DataFrame(index=z_scores_df.index, columns=range(1, n + 1))
    for index in z_scores_df.index:
        # sort the z-scores
        sorted_z_scores = z_scores_df.loc[index].sort_values(ascending=False)
        # get the top n z-scores tickers
        tickers = sorted_z_scores.index[:n]
        # add to the top n DataFrame
        top_n.loc[index] = tickers
    return top_n 
    

def _read_single_alpha_file(file: Path) -> dict[str, pd.DataFrame]:
    """
    Read a single alpha file
    :param file: path to the file
    :return: Dictionary with the file name as key and the DataFrame as value
    """
    # read the file
    df = pd.read_csv(file, index_col=0, parse_dates=True)[OVERALL_SIGNIFICANT_ALPHAS]
    return {file.stem: df}


def read_all_alphas_values(path: Path = Path("output/alphas"),
                           max_workers: int = 10) -> dict[str, pd.DataFrame]:
    """
    Read all alphas values the path
    :param path: path to the folder
    :param max_workers: number of workers to use
    :return: DataFrame with alphas values, index date, columns tickers
    """
    # get all files in the folder
    files = list(path.glob("*.csv"))
    index_cols = [0] * len(files)
    parse_dates = [True] * len(files)
    # read all files and concatenate them into a single DataFrame
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # read all files in parallel
        alphas_values = list(executor.map(_read_single_alpha_file, files))
    # add them into a single dict
    alphas_values_dict = {}
    for alpha in alphas_values:
        alphas_values_dict.update(alpha)

    return alphas_values_dict
    


def single_stock_time_series(stock_returns: pd.Series,
                             alphas_value: pd.DataFrame,
                             month_end: pd.Timestamp,
                             next_month_end: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """
    At time t:
    using previous 3 months of alpha value and stock returns
    
    OLS:
    1. Fit the OLS model
    2. Find the significant alphas
    
    ARIMA:
    1. Find optimal p, q for ARIMA model + alphas value
    2. Fit the model
    3. predict 1 day ahead
    4. get the residuals

    GARCH:
    1. Using the residuals from ARIMA model
    2. Find optimal p, q for GARCH model
    3. Fit the model
    4. predict 1 day volatility

    Then, at time t+1:
    using same p, q for ARIMA and GARCH model
    1. Fit the ARIMA
    2. Predict 1 day ahead
    3. Get the residuals
    4. Fit the GARCH model
    5. Predict 1 day volatility

    Rolling fit until next month
    
    :param stock_returns: Series with stock returns, index date, values returns
    :param alphas_value: DataFrame with alphas values, index date, columns tickers
    :param month_end: month end date
    :param next_month_end: next month end date
    :return: tuple with the expected_returns and volatility predictions, pd.Series
    index = date, values = returns and volatility
    """
    # print(f"=============================\n Processing {month_end.strftime('%Y-%m-%d')} {stock_returns.name} ...")
    # 0. Set data period, 3 months before to next month end
    data_start_date = month_end - pd.DateOffset(months=3) 

    stock_returns_ = stock_returns.loc[data_start_date: next_month_end].copy()
    alphas_value_ = alphas_value.loc[data_start_date: next_month_end].copy()
    look_back_length: int = stock_returns.loc[data_start_date: month_end].shape[0]
    
    # 1. OLS find significant alphas
    significant_alphas = _ols_model_find_significant_alphas(stock_returns_[:month_end], alphas_value_[:month_end])
    alphas_value_: pd.DataFrame = alphas_value_[significant_alphas].copy()

    # 2. ARIMA find optimal p, q
    p_ARMA, q_ARMA = _optimal_p_q_arima(stock_returns_[:month_end], alphas_value_[:month_end])
    if alphas_value_.empty:
        resids = ARIMA(stock_returns_[:month_end], order=(p_ARMA, 0, q_ARMA)).fit().resid
    else:
        resids = ARIMA(stock_returns_[:month_end], order=(p_ARMA, 0, q_ARMA), exog=alphas_value_[:month_end]).fit().resid
    # 3. GARCH find optimal p, q
    p_GARCH, q_GARCH = _optimal_p_q_garch(resids)

    expected_returns = pd.Series(index=stock_returns_.loc[month_end:].index, dtype=float)
    volatility = pd.Series(index=stock_returns_.loc[month_end:].index, dtype=float)
    for date in stock_returns_.loc[month_end: next_month_end].index:
        # setup rolling data
        current_index: int = stock_returns_.index.get_loc(date)  # type: ignore
        start_index = current_index - look_back_length + 1
        rolling_stock_returns = stock_returns_.iloc[start_index: current_index + 1].copy()
        rolling_alphas_value = alphas_value_.iloc[start_index: current_index + 1].copy()

        # 4. Fit the ARIMA model and predict 1 day ahead
        expected_returns[date], residuals = _fit_arima_and_predict(rolling_stock_returns, rolling_alphas_value, p_ARMA, q_ARMA)

        # 5. Fit the GARCH model and predict 1 day ahead
        volatility[date] = _fit_garch_and_predict(residuals, p_GARCH, q_GARCH)

    expected_returns.name = stock_returns.name
    volatility.name = stock_returns.name
    return expected_returns.shift(1).dropna(), volatility.shift(1).dropna()


def _ols_model_find_significant_alphas(stock_returns: pd.Series,
                                        alphas_value: pd.DataFrame) -> list[str]:
    """
    Fit the OLS model and find the significant alphas
    :param stock_returns: Series with stock returns, index date, values returns
    :param alphas_value: DataFrame with alphas values, index date, columns tickers
    :return: list of significant alphas names
    """
    if alphas_value.empty:
        return []
    # shift the alphas value to t-1
    alphas_t_minus_1 = alphas_value.shift(1).copy().dropna(how="all", axis=0).dropna(axis=1).ffill()
    y = stock_returns.loc[alphas_t_minus_1.index].copy().ffill()
    
    # fit the OLS model
    model = sm.OLS(y, alphas_t_minus_1).fit()
    # get the significant alphas
    return model.pvalues[model.pvalues < 0.05].index.tolist()
        


def _optimal_p_q_arima(stock_returns: pd.Series,
                       alphas_value: pd.DataFrame) -> tuple[int, int]:
    """
    Find the optimal p, q for ARIMA model using AIC
    :param stock_returns: Series with stock returns, index date, values returns
    :param alphas_value: DataFrame with alphas values, index date, columns tickers
    :return: tuple with p, q
    """
    if alphas_value.empty:
        y = stock_returns.copy()
    else:
        # shift the alphas value to t-1
        alphas_t_minus_1 = alphas_value.shift(1).copy().dropna(how="all", axis=0).dropna(how="all", axis=1)
        y = stock_returns.loc[alphas_t_minus_1.index].copy()

    AICs = []
    for p in range(1, 4):
        for q in range(1, 4):
            try:
                # fit the ARIMA model
                if  alphas_value.empty:
                    model = ARIMA(y, order=(p, 0, q)).fit()
                else:
                    model = ARIMA(y, order=(p, 0, q), exog=alphas_t_minus_1).fit()
                # get the AIC
                AICs.append((p, q, model.aic))
            except Exception as e:
                print(f"Error fitting ARIMA model: {e}")
                continue
    # get the optimal p, q
    optimal_p, optimal_q = min(AICs, key=lambda x: x[2])[:2]
    return optimal_p, optimal_q


def _fit_garch_and_predict(residuals: pd.Series,
                            p: int,
                            q: int) -> float:
    """
    Fit the GARCH model and predict 1 day ahead
    :param residuals: Series with residuals, index date, values residuals
    :param p: GARCH p parameter
    :param q: GARCH q parameter
    :return: float with the volatility prediction
    """
    # fit the GARCH model
    model = arch_model(residuals, vol="GARCH", p=p, q=q).fit(disp="off")
    
    # predict 1 day ahead
    volatility = model.forecast(horizon=1).variance.values[-1, -1]
    return volatility ** 0.5


def _fit_arima_and_predict(stock_returns: pd.Series,
                           alphas_value: pd.DataFrame,
                           p: int,
                           q: int) -> tuple[float, pd.Series]:
    """
    Fit the ARIMA model and predict 1 day ahead
    :param stock_returns: Series with stock returns, index date, values returns
    :param alphas_value: DataFrame with alphas values, index date, columns tickers
    :param p: ARIMA p parameter
    :param q: ARIMA q parameter
    :return: tuple with the expected_returns and volatility predictions, pd.Series
    index = date, values = returns and volatility
    """
    if alphas_value.empty:
        y = stock_returns.copy()
        model = ARIMA(y, order=(p, 0, q)).fit()
        expected_return = model.forecast(steps=1)
    else:
        # shift the alphas value to t-1
        alphas_t_minus_1 = alphas_value.shift(1).copy().dropna(how="all", axis=0).dropna(how="all", axis=1)
        y = stock_returns.loc[alphas_t_minus_1.index].copy()
        alphas_at_t = alphas_value.iloc[-1].copy()
        # fit the ARIMA model
        model = ARIMA(y, order=(p, 0, q), exog=alphas_t_minus_1).fit()
        
        # predict 1 day ahead
        expected_return = model.forecast(steps=1, exog=alphas_at_t)
    return expected_return, model.resid


def _optimal_p_q_garch(residuals: pd.Series) -> tuple[int, int]:
    """
    Find the optimal p, q for GARCH model using AIC
    :param residuals: Series with residuals, index date, values residuals
    :return: tuple with p, q
    """
    AICs = []
    for p in range(1, 4):
        for q in range(1, 4):
            try:
                # fit the GARCH model
                model = arch_model(residuals, vol="GARCH", p=p, q=q).fit(disp="off")
                # get the AIC
                AICs.append((p, q, model.aic))
            except Exception as e:
                print(f"Error fitting GARCH model: {e}")
                continue
    # get the optimal p, q
    optimal_p, optimal_q = min(AICs, key=lambda x: x[2])[:2]
    return optimal_p, optimal_q


def _single_month_end(month_end: pd.Timestamp,
                      next_month_end: pd.Timestamp,
                      stock_returns: pd.DataFrame,
                      alphas_value_dict: dict[str, pd.DataFrame],
                      max_workers: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a single month end, get the expected returns and volatility
    :param month_end: month end date
    :param stock_returns: DataFrame with stock returns, index date, columns tickers
    :param alphas_value_dict: Dictionary with the file name as key and the DataFrame as value
    :param max_workers: number of workers to use
    :return: tuple with the expected_returns and volatility predictions, pd.Series
    index = date, values = returns and volatility
    """
    # get the expected returns and volatility for each ticker
    expected_returns = []
    volatility = []
    
    stock_list = stock_returns.columns.tolist()
    stock_returns_list = [stock_returns[ticker].copy() for ticker in stock_list]
    alphas_value_list = [alphas_value_dict[ticker].copy() for ticker in stock_list]
    month_end_list = [month_end] * len(stock_list)
    next_month_end_list = [next_month_end] * len(stock_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # fit the model for each stock in parallel
        results = list(executor.map(single_stock_time_series, stock_returns_list, alphas_value_list, month_end_list, next_month_end_list))
        for expected_return, vol in results:
            expected_returns.append(expected_return)
            volatility.append(vol)
    # for ticker in stock_list:
    #     print(ticker)
    #     expected_return, vol = single_stock_time_series(stock_returns[ticker].copy(), alphas_value_dict[ticker].copy(), month_end, next_month_end)
    #     expected_returns.append(expected_return)
    #     volatility.append(vol)
        
    return pd.concat(expected_returns, axis=1), pd.concat(volatility, axis=1)


def predict_returns_volatility(start: pd.Timestamp,
                               top_n: int = 10,
                               output_path: Path = Path("output/predictions")) -> None:
    """
    :param start: start date for prediction
    :param top_n: number of top z-scores to predict
    :output_path: path to save the predictions
    """
    # read the data
    z_scores_df: pd.DataFrame = pd.read_csv("output/z_scores/z_scores_9.csv", index_col=0, parse_dates=True)
    stock_returns: pd.DataFrame = pd.read_csv("output/stock_returns.csv", index_col=0, parse_dates=True)
    top_n_tickers: pd.DataFrame = sort_top_n_z_score(z_scores_df, n=top_n).loc[start:]
    alphas_value_dict: dict[str, pd.DataFrame] = read_all_alphas_values()

    for month_end in top_n_tickers.index[:-1]:
        print(f"=============================\n Processing {month_end.strftime('%Y-%m-%d')} ...")
        # get the next month end
        next_month_end: pd.Timestamp = top_n_tickers.index[top_n_tickers.index.get_loc(month_end) + 1]  # type: ignore
        # get the top n tickers for the month end
        top_n_tickers_ = top_n_tickers.loc[month_end].to_list()
        single_month_stock_returns = stock_returns[top_n_tickers_].copy()
        single_month_alphas_value_dict: dict[str, pd.DataFrame] = {ticker: alphas_value_dict[ticker] for ticker in top_n_tickers_}
        # get the expected returns and volatility
        expected_return, volatility = _single_month_end(month_end, next_month_end, single_month_stock_returns, single_month_alphas_value_dict)

        # save the results
        # create the output folder if not exists
        (output_path / "expected_returns").mkdir(parents=True, exist_ok=True)
        (output_path / "volatilities").mkdir(parents=True, exist_ok=True)
        expected_return.to_csv(output_path / "expected_returns" / f"{month_end.strftime('%Y-%m-%d')}.csv")
        volatility.to_csv(output_path / "volatilities" / f"{month_end.strftime('%Y-%m-%d')}.csv")
