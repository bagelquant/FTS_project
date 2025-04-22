"""calculate factor"""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

def _get_all_symbols(path: Path = Path('data/raw')) -> list[str]:
    return [f.stem for f in path.glob('*.csv')]

def _get_all_alpha_names(path: Path = Path('data/alphas_by_name')) -> list[str]:
    all_alphas = [f.stem for f in path.glob('*.csv')]

    # Remove some alphas that difficult to analyze
    remove_list = ['open',
                   'high',
                   'low',
                   'close',
                   'vol',
                   'pct_change',
                   'vwap',
                   'alpha004',
                   'alpha007',
                   'alpha019',
                   'alpha021',
                   'alpha027',
                   'alpha036',
                   'alpha037',
                   'alpha052',
                   'alpha061',
                   'alpha062',
                   'alpha064',
                   'alpha065',
                   'alpha068',
                   'alpha074',
                   'alpha075',
                   'alpha078',
                   'alpha081',
                   'alpha094',
                   'alpha095',
                   'alpha099']

    for alpha in remove_list:
        if alpha in all_alphas:
            all_alphas.remove(alpha)
    return all_alphas


def _read_alpha(alpha_name: str, 
               path: Path = Path('data/alphas_by_name')) -> pd.DataFrame:
    """read alpha"""
    return pd.read_csv(path / f'{alpha_name}.csv', index_col=0, parse_dates=True)


@dataclass(slots=True)
class Factor:
    
    alpha_name: str
    pct_change: pd.DataFrame  # percentage change of stocks price
    alpha: pd.DataFrame  # alpha value of stocks
    groups: int = 5
    
    def daily_group(self, date: datetime) -> pd.Series:
        """
        Sort alpha by date,
        then divide into groups,

        each group calculated equal weight pct_change,
        return a series of pct_change(groups)
        """
        # sort the alpha
        sorted = self.alpha.loc[date].sort_values(ascending=True)  # type: ignore

        # divide into groups -> {group: [stocks]}
        group = {i: sorted.iloc[i::self.groups].index for i in range(self.groups)}

        # calculate equal weight pct_change
        pct_change = {i: self.pct_change.loc[date, group[i]].mean() for i in group}   # type: ignore
        return pd.Series(pct_change, name=date)

    def group_return(self) -> pd.DataFrame:
        """
        Calculate group return from start to end
        """
        return pd.concat([self.daily_group(date) for date in self.alpha.index], axis=1).T

    def hml(self) -> pd.Series:
        """
        Calculate HML
        """
        # calculate group return
        group_return = self.group_return()

        # calculate HML
        hml = group_return[self.groups-1] - group_return[0]
        hml.name = self.alpha_name
        return hml


def calculate_hmls() -> None:
    """concat all factors"""
    all_alphas = _get_all_alpha_names()
    
    # calculate hml
    pct_change = _read_alpha('pct_change')/100
    hmls = []
    for alpha in all_alphas:
        alpha_df = _read_alpha(alpha)
        factor = Factor(alpha, pct_change, alpha_df)
        hml = factor.hml()
        hmls.append(hml)
    hmls = pd.concat(hmls, axis=1)

    eqw = pct_change.mean(axis=1)
    hmls['eqw'] = eqw
    hmls['AAPL'] = pct_change['AAPL']

    # change columns order by name
    hmls = hmls[sorted(hmls.columns)]
    hmls.to_csv('data/hml.csv')
    print("HMLs calculated and saved to data/hml.csv")


if __name__ == '__main__':
    from time import perf_counter
    start = perf_counter()
    calculate_hmls()
    print(f"\nTime: {perf_counter() - start:.2f}s")
