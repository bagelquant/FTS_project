
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from scipy.stats import jarque_bera, shapiro
import os
import numdifftools as nd


def return_process(return_data: pd.DataFrame) -> pd.DataFrame:
    """
    We need monthly returns at the end of each month

    :param return_data: DataFrame with date as index, stocks as columns and close price as values
    """

    return_data.index = pd.to_datetime(return_data.index, format='%Y%m%d')
    return_data = return_data.sort_index()
    return_data = return_data.ffill()
    last_ = return_data.resample('ME').last()
    first_ = return_data.resample('ME').first()
    last_ = last_.sort_index()
    first_ = first_.sort_index()
    monthly_return = (last_ - first_) / first_  
    
    monthly_return = monthly_return.dropna(axis=1, how='any')
    return monthly_return

def match_data(returns: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the data of returns and factors

    :param returns: DataFrame with date as index, stocks as columns and close price as values
    :param factors_df: DataFrame with date as index, factors as columns and factor values as values
    :return: DataFrame with date as index, stocks as columns and close price as values
    """
    returns.index = returns.index + pd.offsets.MonthEnd(0)
    factors_df.index = factors_df.index + pd.offsets.MonthEnd(0)
    returns = returns.loc[(returns.index >= factors_df.index[0]) & (returns.index <= factors_df.index[-1])]
    factors_df = factors_df.loc[(factors_df.index >= returns.index[0]) & (factors_df.index <= returns.index[-1])]

    returns, factors_df = returns.align(factors_df, join='outer', axis=0)
    returns = returns.ffill()
    factors_df = factors_df.ffill()
    returns = returns.dropna()
    factors_df = factors_df.dropna()
    return returns, factors_df


def regression(returns: pd.Series, factors: pd.DataFrame) -> tuple:
    """
    Perform regression of returns on factors

    :param returns: (n, ) Series with date as index and stock returns as values
    :param factors: (n, k) DataFrame with date as index and factors as columns
    :return: tuple of regression results
    """
    X = np.column_stack((np.ones(len(factors)), factors))
    y = returns.values
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta
    return beta, residuals


def residual_analysis(residuals: np.ndarray, lags=20) -> pd.DataFrame:
    """
    Perform residual analysis

    :param residuals: array of residuals
    :return: tuple of mean, variance and skewness
    """
    # autocorrelation test
    lb_result = acorr_ljungbox(residuals, lags=[lags], return_df=True).iloc[0]
    
    # ARCH test
    arch_test = acorr_ljungbox(residuals**2, lags=[lags], return_df=True).iloc[0]

    # normality test
    shap = shapiro(residuals)
    jb = jarque_bera(residuals)

    result_df = pd.DataFrame({
        'Test': ['Ljung-Box', 'ARCH', 'Shapiro-Wilk', 'Jarque-Bera'],
        'Stat': [lb_result['lb_stat'], arch_test['lb_stat'], shap[0], jb[0]],
        'p-value': [lb_result['lb_pvalue'], arch_test['lb_pvalue'], shap[1], jb[1]]
    })
    return result_df


def garch_order(residuals: np.ndarray) -> tuple:
    """
    GARCH order selection using AIC
    
    """
    
    best_aic = np.inf
    
    best_order = (0, 0)
    for p in range(1, 3):
        for q in range(0, 3):
            if p == 0 and q == 0:
                continue
            model = arch_model(residuals, vol='Garch', p=p, q=q, o=0)
            results = model.fit(disp='off')
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, q)
    return best_order

def GARCH_M_FUNC(params: np.array, returns: pd.Series, factors: pd.DataFrame, p: int=1, q: int=1) -> pd.DataFrame:
    """
    GARCH-M function
    """
    n = len(returns)
    k = factors.shape[1]

    alpha = params[0]
    beta = params[1:k+1]
    lambd = params[k+1]
    omega = params[k+2]
    a = params[k+3: k+3+p]
    b = params[k+3+p: k+3+p+q]
    if sum(a) + sum(b) > 0.99:
        return None, None
    
    # todo:  simply fill the first p and q values which are not accurate.
    eps = np.zeros(n)
    mean_return = returns.mean()
    for t in range(max(p, q)):
        eps[t] = returns[t] - mean_return
    h = np.ones(n) * np.var(returns)
    for t in range(max(p,q), n):
        arch_sum = np.sum([a[i] * eps[t - i - 1]**2 for i in range(p)]) if p > 0 else 0
        garch_sum = np.sum([b[j] * h[t - j - 1] for j in range(q)]) if q > 0 else 0
        h[t] = omega + arch_sum + garch_sum
        if h[t] <= 0:
            h[t] = 1e-6
        if k < 1:
            eps[t] = returns[t] - alpha - lambd * np.sqrt(h[t])
        else:
            eps[t] = returns[t] - alpha - np.dot(beta, factors.iloc[t]) - lambd * np.sqrt(h[t])
        eps[t] = np.clip(eps[t], -1e6, 1e6) 
    
    eps = eps[max(p,q):]
    h = h[max(p,q):]

    return eps, h


def garch_m_likelihood(params: np.ndarray, returns: pd.Series, factors: pd.DataFrame, p: int=1, q: int=1) -> float:
    """
    GARCH-M likelihood function
    """
    eps, h = GARCH_M_FUNC(params, returns, factors, p, q)
    # add punishment to make sure the model is stationary
    if eps is None:
        return 1e6
    loglik = 0
    for t in range(0, len(eps)):
        eps_sq = eps[t]**2
        eps_sq_div_ht = eps_sq / max(h[t], 1e-6)   
        eps_sq_div_ht = min(eps_sq_div_ht, 1e6)           
        log_ht = np.log(max(h[t], 1e-6))                   

        loglik += 0.5 * (log_ht + eps_sq_div_ht)
    return loglik


def GARCHM(returns: pd.Series, factors_df: pd.DataFrame, p: int = 1, q: int = 1) -> pd.DataFrame:
    """
    GARCH-M model with multiple factors

    :param returns: DataFrame with date as index, stocks as columns and close price as values
    :param factors_df: DataFrame with date as index, factors as columns and factor values as values
    :return:
    """

    k = factors_df.shape[1]
    garch_params = [0.1] + [0.1/p if p != 0 else 0] * p + [0.7/(q) if q != 0 else 0] * q
    mean_return = returns.mean()
   
    init_params = np.array([mean_return] + [0.1]*k + [0.1] + garch_params)
    bounds = [(None, None)] + [(None, None)] * k + [(None, None)] + [(0, None)] + [(0, None)] * (p + q)
    
    # GARCH-M model
    res = minimize(
        garch_m_likelihood,
        init_params,
        args=(returns, factors_df, p, q),
        bounds=bounds,
        method='SLSQP',
        options={'disp': True}
    )
    params = res.x

    # Significance test
    try:
        hess_fn = nd.Hessian(lambda p_: garch_m_likelihood(p_, returns, factors_df, p, q))
        hessian = hess_fn(params)
    
        if not np.all(np.isfinite(hessian)):
            raise ValueError("Hessian contains nan or inf.")

        try:
            cov_matrix = np.linalg.pinv(hessian)
        except np.linalg.LinAlgError:
            raise ValueError("Hessian is too singular for pinv.")

        std_errors = np.sqrt(np.diag(cov_matrix))
    
    except Exception as e:
        print("Switching to fallback: eigenvalue-regularized Hessian")
        hess_fn = nd.Hessian(lambda p_: garch_m_likelihood(p_, returns, factors_df, p, q))
        hessian = hess_fn(params)
        eigval, eigvec = np.linalg.eigh(hessian)
        eigval[eigval < 1e-6] = 1e-6
        hessian_reg = eigvec @ np.diag(eigval) @ eigvec.T

        cov_matrix = np.linalg.inv(hessian_reg)
        std_errors = np.sqrt(np.diag(cov_matrix))
    
    std_errors = np.clip(std_errors, 1e-6, 1e6)
    t_stats = params / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

    sector = ['intercept'] + ['factor'] * k + ['garch_m'] +\
             ['garch'] * (p + q + 1)
    # Print results
    param_names = (
    ['alpha'] + factors_df.columns.tolist() +
    ['lambda', 'omega'] +
    [f'a_{i+1}' for i in range(p)] +
    [f'b_{j+1}' for j in range(q)]

)
    
    results_df = pd.DataFrame({
        'param': param_names,
        'value': params,
        'std_error': std_errors,
        't_stat': t_stats,
        'p_value': p_values,
        'sector': sector
    })

    eps, h = GARCH_M_FUNC(params, returns, factors_df, p, q)
    if eps is None:
        print("Model is not stationary.")
        return None, None
    else:
        # residual analysis
        print("Residual analysis...")
        standardized_residuals = np.clip(eps / np.sqrt(h), -1e6, 1e6)
        dignose_result = residual_analysis(standardized_residuals)

    return results_df, dignose_result 

def read_data():
    """
    read stocks returns and factors data
    """

    factors = pd.read_csv('../data/rolling_result_p.csv', index_col=0, header=0)
    factors.index = pd.to_datetime(factors.index, format='%Y-%m-%d')

    # return of stock
    df_returns = pd.DataFrame()
    stocks = []
    for dirpath, dirnames, filenames in os.walk('../data/raw/'):
        for filename in filenames:
            if filename.endswith('.csv'):
                stocks.append(filename.split('.')[0])
                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path, index_col=0, header=0)
                df.index = pd.to_datetime(df.index, format='%Y%m%d')
                df = df[['close']]
                df_returns = pd.concat([df_returns, df], axis=1, join='outer')
    df_returns.columns = stocks
    monthly_return = return_process(df_returns)


    # match the data
    monthly_return, factors_df = match_data(monthly_return, factors)
    return monthly_return, factors_df

def pre_training(return_: pd.Series, factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the data before training and decide order of GARCH model
    """

    # regression
    # beta, residuals = regression(return_, factors_df)

    # residual analysis
    test_result = residual_analysis(return_)

    # GARCH order selection
    if test_result[test_result['Test'] == 'ARCH']['p-value'].values[0] > 0.1:
        return 0, 0, test_result
    p, q = garch_order(return_)
    print(f"Best GARCH order: {p}, {q}")
    return p, q, test_result


def training(return_: pd.Series, factors_df: pd.DataFrame, 
             p: int=1, q: int=1, backward=True): 
    """
    Training function
    :param returns: DataFrame with date as index, stocks as columns and close price as values
    :param factors_df: DataFrame with date as index, factors as columns and factor values as values
    :param p: GARCH order
    :param q: GARCH order
    :param backward: whether to use backward selection. True, delete the insignificant factors and rerun. False, keep all factors

    :return:
    """

    # GARCH-M model
    results_df, diagnose_result = GARCHM(return_, factors_df, p, q)
    if results_df is None:
        return None, None
    if backward:
        # backward selection
        # check factors p-value
        significant_factors = results_df[results_df['p_value'] < 0.05]['param'].tolist()
        significant_factors = [f for f in significant_factors if f in factors_df.columns]
        if len(significant_factors) == len(factors_df.columns):
            pass
        else:
            if len(significant_factors) == 0:
                new_factor_df = pd.DataFrame()
            else:
                new_factor_df = factors_df[significant_factors]

                # rerun the model with significant factors
                results_df, diagnose_result = GARCHM(return_, new_factor_df, p, q)


    return results_df, diagnose_result


def predict(returns: pd.Series, factors: pd.DataFrame, model: pd.Series, p: int=1, q: int=1)-> pd.DataFrame:
    """
    Predict the future returns using GARCH-M model
    todo: can only predict one step ahead in our project
    :param returns: (n, 1)
    :param factors: (n + 1, k)
    """
    model = model.values
    # get residuals and volatility
    eps, h = GARCH_M_FUNC(model, returns, factors.iloc[:-1], p, q)
    # predict
    k = factors.shape[1]
    alpha = model[0]
    beta = model[1:k + 1]
    lambd = model[k + 1]
    omega = model[k + 2]
    a = model[k + 3: k + 3 + p]
    b = model[k + 3 + p: k + 3 + p + q]

    arch_sum = np.sum([a[i] * eps[-i - 1] ** 2 for i in range(p)]) if p > 0 else 0
    garch_sum = np.sum([b[j] * h[-j - 1] for j in range(q)]) if q > 0 else 0
    h_next = omega + arch_sum + garch_sum
    h_next = max(h_next, 1e-6)
    
    if k < 1:
        r_next = alpha + lambd * np.sqrt(h_next)
    else:
        r_next = alpha + np.dot(beta, factors.iloc[-1]) + lambd * np.sqrt(h_next)
    return r_next, h_next



def batch_predict(factors_df: pd.DataFrame, returns: pd.DataFrame):
    """
    predict the next month return of stocks
    """
    training_factors = factors_df.iloc[:-1]
    predict_returns = pd.Series(index=returns.columns)
    predict_volatility = pd.Series(index=returns.columns)
    training_info = {}
    for stock in returns.columns:
        print(f"Predicting {stock}...")
        return_ = returns[stock]
        p, q, test_result = pre_training(return_,training_factors) 
        if p == 0 and q == 0:
            print(f"Stock {stock} has no ARCH effect, skip.")
            continue

        model_result, diagnose_result = training(return_, training_factors, p, q, backward=True)
        if model_result is None:
            continue
        else:
            # if the model is not converged, skip
            current_factors = model_result[model_result['sector'] == 'factor']['param'].tolist()
            print("Training finished.")
            # predict
            print("Predicting...")
            new_factor_df = factors_df[current_factors]
            r_next, h_next = predict(return_, new_factor_df, model_result['value'], p, q) 

            predict_returns[stock] = r_next
            predict_volatility[stock] = h_next
            training_info[stock] = {"model": model_result,
                                    "test_result": test_result,
                                    "diagnose_result": diagnose_result}
            print("Predicting finished.")


    return predict_returns, predict_volatility, training_info


def rolling_predict(factors_df: pd.DataFrame, returns: pd.DataFrame, training_period = 24):
    """
    rolling predict the next month return of stocks
    """
    dates = returns.iloc[training_period + 1:].index
    predict_returns = pd.DataFrame(columns=returns.columns, index=dates)
    predict_volatility = pd.DataFrame(columns=returns.columns, index=dates)
    training_infos = {}
    print("Rolling predict...")
    for i in range(len(returns) - training_period - 1):
        next_returns, next_vol, current_training = batch_predict(factors_df.iloc[i:i + training_period + 1], returns.iloc[i + 1:i + training_period + 1])
        predict_returns.iloc[i] = next_returns
        predict_volatility.iloc[i] = next_vol 
        training_infos[dates[i]] = current_training 
        
    return predict_returns, predict_volatility, training_infos

def main(start_date='2020-01-01', end_date='2024-12-31', save_path='../data/results', window=24, pure_garchm=False):
    """
    Main function
    """

    # read data
    returns, factors_df = read_data()
    # data begain date 
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data_start_date = start_date - pd.DateOffset(months=window)

    returns = returns.loc[data_start_date:end_date]
    factors_df = factors_df.loc[data_start_date:end_date]

    if pure_garchm:
        factors_df = pd.DataFrame()


    # rolling predict
    predict_returns, predict_volatility, training_info = rolling_predict(factors_df, returns, training_period=window)
    error_returns = predict_returns - returns.iloc[window + 1:]
    # save results
    predict_returns.to_csv(f'{save_path}/predict_returns.csv')
    predict_volatility.to_csv(f'{save_path}/predict_volatility.csv')
    error_returns.to_csv(f'{save_path}/error_returns.csv')
    

    for date_ in training_info.keys():
        for stock in training_info[date_].keys():
            date_str = date_.strftime('%Y%m%d')
            training_info[date_][stock]['model'].to_csv(f'{save_path}/training_info_{stock}_{date_str}.csv')
            training_info[date_][stock]['test_result'].to_csv(f'{save_path}/test_result_{stock}_{date_str}.csv')
            training_info[date_][stock]['diagnose_result'].to_csv(f'{save_path}/diagnose_result_{stock}_{date_str}.csv')



if __name__ == '__main__':
    main(start_date='2024-11-01', end_date='2024-12-31', save_path='../data/results', window=144, pure_garchm=True)






