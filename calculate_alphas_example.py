"""
Example of calculating alphas

Data:
- raw data: price and volume data of SP500 stocks
- alphas: calculated from the raw data, stored in `data/alphas` folder, each symbol has its own file
- concat

"""
from time import perf_counter
from src.data_module import calculate_all_alphas, regroup_alphas

if __name__ == '__main__':
    print("Start calculating all alphas")
    start = perf_counter()
    calculate_all_alphas()
    end = perf_counter()
    print(f"Calculate all alphas: {end - start:.2f} seconds or {(end - start) / 60:.2f} minutes")
    
    print("Start regrouping alphas")
    start = perf_counter()
    regroup_alphas()
    end = perf_counter()
    print(f"Regroup all alphas: {end - start:.2f} seconds or {(end - start) / 60:.2f} minutes")
