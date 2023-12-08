import time

def getNowFormatTime() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

print(getNowFormatTime())

