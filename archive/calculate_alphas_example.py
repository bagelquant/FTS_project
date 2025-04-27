"""
Example of calculating alphas

Data:
- raw_data data: price and volume data of SP500 stocks
- alphas_by_symbol: calculated from the raw_data data, stored in `data/alphas` folder, each symbol has its own file
- alphas_by_name: regrouped alphas, stored in `data/alphas_by_name` folder, each alpha has its own file

hml.csv:
- run `calculate_hmls` to calculate HML portfolio percentage change for each alpha
- stored in `data/hml.csv` file
"""
from time import perf_counter
from src.data_module import calculate_all_alphas, regroup_alphas
from src.data_module import calculate_hmls

if __name__ == '__main__':
    # print("Start calculating all alphas")
    # start = perf_counter()
    # calculate_all_alphas()
    # end = perf_counter()
    # print(f"Calculate all alphas: {end - start:.2f} seconds or {(end - start) / 60:.2f} minutes")
    
    # print("Start regrouping alphas")
    # start = perf_counter()
    # regroup_alphas()
    # end = perf_counter()
    # print(f"Regroup all alphas: {end - start:.2f} seconds or {(end - start) / 60:.2f} minutes")

    print("Start calculating HMLs")
    start = perf_counter()
    calculate_hmls()
    end = perf_counter()
    print(f"Calculate HMLs: {end - start:.2f} seconds or {(end - start) / 60:.2f} minutes")
