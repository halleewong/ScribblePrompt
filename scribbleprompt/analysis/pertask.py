import pandas as pd
from pylot.util.thunder import ThunderReader

def load_epoch_task_stats(path):
    r = ThunderReader(path)
    dfs = []
    for k in r:
        _, phase, epoch = k.split(".")
        df = r[k]
        #df = df.groupby(["task","example", "iter"], as_index=False).mean()
        df["phase"] = phase
        df["epoch"] = int(epoch)
        df["path"] = path
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)