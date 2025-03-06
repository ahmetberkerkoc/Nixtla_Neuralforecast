import pandas as pd
def gas_demand():
    df = pd.read_csv("data/turkey_gas.csv")
    df = df.drop("Unnamed: 0", axis=1)
    df["y-1"] = df["ABONE"].shift(1)
    df["y-2"] = df["ABONE"].shift(2)
    df["y-3"] = df["ABONE"].shift(3)
    df = df.dropna()
    df.reset_index(inplace=True, drop=True)
    df["unique_id"] = range(1,len(df)+1)
    df['ds'] = pd.date_range(start="2000-01-01 00:00:00", periods=len(df), freq="D")


    test_size = int(len(df) * 0.1)
    return df, test_size
