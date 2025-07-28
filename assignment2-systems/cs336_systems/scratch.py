import numpy as np
import pandas as pd

records = []
arr = []

size, n_procs, backend = None, None, None
with open("cs336_systems/tmp.txt", "r") as f:
    while True:
        line = f.readline()
        if line is None or len(line.strip()) == 0:
            break

        tmp = line.split(" ")
        if tmp[0] == "mean":
            arr.append(float(tmp[-1]))
        else:
            if len(arr) > 0:
                mean_time = np.array(arr).mean()
                record = {
                    "size": size,
                    "n_procs": n_procs,
                    "backend": backend,
                    "mean_time": mean_time
                }
                records.append(record)

            size = float(tmp[4])
            n_procs = float(tmp[9])
            backend = tmp[-1].strip()

            arr = []

if len(arr) > 0:
    mean_time = np.array(arr).mean()
    record = {
        "size": size,
        "n_procs": n_procs,
        "backend": backend,
        "mean_time": mean_time
    }
    records.append(record)

df = pd.DataFrame(records)
print(df.to_latex(index=False))
print(arr)
