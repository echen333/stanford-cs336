import numpy as np

arr = []
with open("cs336_systems/tmp.txt", "r") as f:
    line = f.readline()
    tmp = line.split(" ")
    arr.append(float(tmp[-1]))
print(arr)
