import datetime
import time


def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')