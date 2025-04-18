"""
Read seperate csv file, concat to a table

index: - trade_date: datetime
columes: - symbol: str, from csv file name without extension
"""

import pandas as pd
from pathlib import Path
from .alpha import get_alpha


def _get_stock_price(symbol: str, 
                    path: Path = Path('data/raw')) -> pd.DataFrame:
    """
    Read stock price from csv file
    :param symbol: Stock symbol
    :param path: Path to data folder
    :return: DataFrame with stock price
    
    DataFrame index: trade_date
    DataFrame columns: open, high, low, close, volume
    """
    df = pd.read_csv(path / f'{symbol}.csv', 
                     index_col='trade_date', 
                     parse_dates=True).sort_index()
    return df


def _get_all_symbols(path: Path = Path('data/raw')) -> list[str]:
    """
    Get all symbols(File name) from data folder
    :param path: Path to data folder
    :return: List of symbols
    """
    return [f.stem for f in path.glob('*.csv')]


def _calculate_alphas_single_symbol(symbol: str, 
                                   output_path: Path = Path('data/alphas_by_symbol')) -> None:
    """
    Calculate alpha for a single symbol
    :param symbol: Stock symbol     
    :param output_path: Path to output folder
    :return: None

    Utilize get_alpha function from alpha.py module,
    then save to a csv file
    """
    if not output_path.exists():
        output_path.mkdir()
    df = _get_stock_price(symbol)
    alphas = get_alpha(df)  # Calculate alphas, from alpha.py module
    alphas.to_csv(output_path / f'{symbol}.csv')


def calculate_all_alphas(output_path: Path = Path('data/alphas_by_symbol')) -> None:
    for symbol in _get_all_symbols():
        print(f"Calculate alphas for {symbol} ...")
        _calculate_alphas_single_symbol(symbol, output_path)


def regroup_alphas(path: Path = Path('data/alphas_by_symbol'), 
                   output_path: Path = Path('data/alphas_by_name')) -> None:
    """
    Read all data from alpha folder, 
    regroup the data to `alpha_name.csv`, containing all symbols

    :param path: Path to alpha folder
    :param direction_path: Path to output folder
    :return: None
    """
    if not output_path.exists():
        output_path.mkdir()
    
    columns = pd.read_csv(path / 'AAPL.csv', index_col='trade_date').columns.to_list()
    data = {}
    for symbol in _get_all_symbols(path):
        df = pd.read_csv(path / f'{symbol}.csv', index_col='trade_date', parse_dates=True)
        data[symbol] = df
    for column in columns:
        concat_table = pd.DataFrame({symbol: data[symbol][column] for symbol in data})
        concat_table.to_csv(output_path / f'{column}.csv')


if __name__ == '__main__':
    from time import perf_counter
    start = perf_counter()
    calculate_all_alphas()
    regroup_alphas()
    print(f"Time: {perf_counter() - start:.2f} seconds \n or {(perf_counter() - start) / 60:.2f} minutes")

