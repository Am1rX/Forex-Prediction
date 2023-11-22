import pandas as pd
file_name = input("Enter Yout CSV file to be prepared for prediction [it should be from metatrader history data ! ] >> ")
df = pd.read_csv(file_name, header=None)
df[0] = df[0].astype(str) + " " + df[1].astype(str)
df = df.drop(columns=[6])
df = df.drop(columns=[1])
df[0] = pd.to_datetime(df[0], format='%Y.%m.%d %H:%M')
df[0] = df[0].dt.strftime('%m/%d/%Y %I:%M:%S %p')
df.to_csv("str4n5er.csv", index=False, header=None)
