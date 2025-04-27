"""
Using high-low portfolio returns to test alphas have significant returns compared to zero.

1. Regroup alphas from ticker to alphas, each table
2. Sort alphas by their value
3. Calculate high-low portfolio returns for each alpha -> a dataframe: index: date, columns: alphas_returns
4. t-test
"""

#
# def regroup_alphas(path: Path = Path('data/alphas_by_symbol'),
#                    output_path: Path = Path('data/alphas_by_name')) -> None:
#     """
#     Read all data from alpha folder,
#     regroup the data to `alpha_name.csv`, containing all symbols
#
#     :param path: Path to alpha folder
#     :param output_path: Path to output folder
#     :return: None
#     """
#     if not output_path.exists():
#         output_path.mkdir()
#
#     columns = pd.read_csv(path / 'AAPL.csv', index_col=0).columns.to_list()
#     data = {}
#     for symbol in _get_all_symbols(path):
#         df = pd.read_csv(path / f'{symbol}.csv', index_col=0, parse_dates=True)
#         # remove duplicate index
#         df = df[~df.index.duplicated(keep='first')]
#         data[symbol] = df
#     for column in columns:
#         print(f"Regroup {column} ...")
#         concat_table = pd.DataFrame({symbol: data[symbol][column] for symbol in data})
#         concat_table.to_csv(output_path / f'{column}.csv')

def _test() -> None:
    """Quick test for this module"""
    pass


if __name__ == "__main__":
    from time import perf_counter

    start = perf_counter()
    _test()
    end = perf_counter()
    print(f"Time cost: {end - start:.2f} s \n or {(end - start) / 60:.2f} min")
