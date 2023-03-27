import pandas as pd

df = pd.read_csv('/home/toefl/K/asl-signs/train.csv')
print(len(df))
df = df[df["participant_id"] != 29302]
print(len(df))