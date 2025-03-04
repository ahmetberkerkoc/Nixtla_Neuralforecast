import pandas as pd
def gas_demand():
    df = pd.read_csv("data/turkey_gas.csv")
    df = df.drop("Unnamed: 0", axis=1)
    df["y-1"] = df["ABONE"].shift(1)
    df["y-2"] = df["ABONE"].shift(2)
    df["y-3"] = df["ABONE"].shift(3)
    df = df.dropna()
    y = df.loc[:, "ABONE"]
    X = df.iloc[:, 1:]

    test_size = int(len(df) * 0.2)
    return X, y, test_size
