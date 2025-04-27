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
2. Test for alphas have significant returns compared to zero.
"""

from pathlib import Path
from time import perf_counter
from src import calculate_all_alphas


# CONFIGURATIONS
MAX_WORKERS = 10  # Number of workers for multiprocessing
ALPHA_LEVEL = 0.05  # Significance level for all tests


def main() -> None:
    """
    Uncomment the steps you want to run

    Steps:

    """

    # step 1: calculate all alphas -> Time cost: about 5 mins
    # print("Start calculating all alphas")
    # calculate_all_alphas(output_path=Path('output/alphas'),
    #                      max_workers=MAX_WORKERS)

    # step 2. test for alphas significant


    print("===All steps are done.===")


if __name__ == "__main__":
    start = perf_counter()
    main()
    end = perf_counter()
    print(f"Time cost: {end - start:.2f} s \n or {(end - start) / 60:.2f} min")
