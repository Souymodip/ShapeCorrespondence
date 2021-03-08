import functools
import time
from os import walk
from datetime import datetime
import torch as th



PATH="./Models/"

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


def save_model(model):
    now = datetime.now()
    compression = str(model.dimension) + "_to_" + str(model.final)
    file_name = "M_" + compression + "_on_D" + str(now.day) + "_H" +str(now.hour)+"_M_" + str(now.minute) +".pt"
    th.save(model, PATH+file_name)


def save_model_state(model):
    def string(a):
        return str(a) if a >= 10 else "0"+str(a)
    now = datetime.now()
    compression = str(model.dimension) + "_to_" + str(model.final)
    file_name = "M_" + compression + "_on_D" + string(now.day) + "_H" + string(now.hour) + "_M" + string(now.minute) + ".pt"
    th.save(model.state_dict(), PATH+file_name)


def load_model_state(model, file):
    model.load_state_dict(th.load(file))
    model.eval()
    return model


def load_last_model(dimension, final, model):
    _, _, files = next(walk(PATH))
    def find(name):
        first = name.split("_on_")[0]
        return first == "M_" + str(dimension) +"_to_" + str(final)

    files = [file for file in files if find(file)]
    if len(files) == 0:
        print(bcolors.WARNING + " No saved model state found!!" + bcolors.ENDC)
        return model, False

    files.sort()
    file = PATH+files[-1]
    model = load_model_state(model, file)
    print(bcolors.BOLD + " Loading model {}".format(files[-1]) + bcolors.ENDC)
    return model, True


def load_model():
    _, _, files = next(walk(PATH))
    if len(files) == 0:
        return None

    files.sort()
    print(bcolors.BOLD + " Loading model {}".format(files[-1]) + bcolors.ENDC)
    model = th.load(PATH+files[-1])
    model.eval()
    return model