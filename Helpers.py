import functools
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def timer(func):
    """Print the run time of a decorated function"""
    @functools.wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        end = time.perf_counter()
        run_time = end - start
        print(bcolors.OKBLUE + f"@Function {func.__name__!r} finished in {run_time:.4f}s"+ bcolors.ENDC)
        return value
    return wrapper_timmer


