"""
Main script for the project.

Author:
- Yanzhong(Eric) Huang
- Qinqin Huang
- Yongyi Tang
- Haoyue TANG

Note:
    - This project involves multiple modules and files.
    - The calculation process may take a long time.
    - Using multiprocessing to speed up the calculation. default max_workers=10.

>! We comment out some steps to speed up the process. If you want to run the full process, please uncomment them. !<

The main script will:

1. Calculate all alphas, stored in `output/alphas` folder. -> about 5 mins
2. Regroup alphas from ticker to alphas, each table -> about 5 mins -> For further analysis
3. Calculate z-scores for each stock, stored in `output/z_scores` folder. -> about 2 mins
4. Predict returns and volatility for each stock, stored in `output/predictions` folder. -> about 10 mins
"""

import pandas as pd
import warnings
from pathlib import Path
from time import perf_counter
from src import calculate_all_alphas, regroup_alphas, calculate_stock_z_scores, predict_returns_volatility
# ConvergenceWarning from sklearn

# Suppress all warnings
warnings.filterwarnings("ignore")

# CONFIGURATIONS
MAX_WORKERS = 10  # Number of workers for multiprocessing
ALPHA_LEVEL = 0.05  # Significance level for all tests
PREDICTION_START_DATE: pd.Timestamp = pd.Timestamp("2015-01-01")  # Start date for prediction
TOP_N: int = 10  # Top N stocks to predict per month


def main() -> None:
    """
    Uncomment the steps you want to run

    Steps:

    """
    print("""
===Running the main script...===
For quick run, we comment out some steps.
If you want to run the full process, please uncomment them.

All Steps:
1. Calculate all alphas, stored in `output/alphas` folder. -> about 5 mins
2. Regroup alphas from ticker to alphas, each table -> about 5 mins -> For further analysis
3. Calculate z-scores for each stock, stored in `output/z_scores` folder. -> about 2 mins
4. Predict returns and volatility for each stock, stored in `output/predictions` folder. -> about 10 mins

Time may vary depending on the number of workers, size of data, and your machine.
    """)

    # step 1: calculate all alphas -> Time cost: about 5 mins
    # print("Start calculating all alphas")
    # calculate_all_alphas(output_path=Path('output/alphas'),
    #                      max_workers=MAX_WORKERS)

    # step 2. Regroup alphas -> Time cost: about 5 mins
    # print("Regrouping all alphas...")
    # regroup_alphas(path=Path("output/alphas"),
    #                output_path=Path('output/alphas_by_name'))


    # step 3: calculate z-scores -> Time cost: about 2 mins
    # print("Calculating z-scores...")
    # # lookback_period: list[int] = [3, 6, 9, 12, 18, 24]  # lookback periods in months
    # lookback_period: list[int] = [9]  # for quick run
    # calculate_stock_z_scores(lookback_periods=lookback_period,
    #                          output_path=Path("output/z_scores"))

    # step 4: time-series prediction -> Time cost: about 10 mins
    # print("Predicting returns and volatility...")
    # predict_returns_volatility(start=PREDICTION_START_DATE,
    #                            top_n=TOP_N,
    #                            output_path=Path("output/predictions"))
    # print("The predictions are saved in output/predictions folder.")
    print("===All steps are done.===")


if __name__ == "__main__":
    start = perf_counter()
    main()
    end = perf_counter()
    print(f"Time cost: {end - start:.2f} s \n or {(end - start) / 60:.2f} min")
