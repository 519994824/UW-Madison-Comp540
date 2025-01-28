import pandas as pd

df = pd.read_csv("history.csv", index_col=False)
df = df.dropna(subset=["Days of Ice Cover"])
df = df.rename(columns={"Winter": "year", "Days of Ice Cover": "days"})

df["year"] = df["year"].str[:4].astype("int")
df["days"] = df["days"].astype("int")

df = df.drop(["Freeze-Over Date", "Thaw Date"], axis=1)

df.to_csv("hw5.csv", index=False)



