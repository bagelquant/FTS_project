"""
<MODULE NAME>

Author: Yanzhong(Eric) Huang
"""


def _test() -> None:
    """Quick test for this module"""
    pass


if __name__ == "__main__":
    from time import perf_counter

    start = perf_counter()
    _test()
    end = perf_counter()
    print(f"Time cost: {end - start:.2f} s \n or {(end - start) / 60:.2f} min")
