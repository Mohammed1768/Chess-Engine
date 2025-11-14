import pandas as pd
import numpy as np

df = pd.read_csv('evaluations.csv')

pos = df[df['eval'] > 0]
neg = df[df['eval'] <= 0]

if len(pos) > len(neg):
    pos = pos.sample(len(neg), random_state=42)
else:
    neg = neg.sample(len(pos), random_state=42)

new_df = pd.concat([pos, neg], ignore_index=True)
new_df['eval'] = np.tanh(new_df['eval']/700)


mates = pd.read_csv('mates.csv')
mates = mates.sample(frac=1, random_state=42).reset_index(drop=True)

mates = mates[:100_000]

new_df = pd.concat([mates, new_df], ignore_index=True)
new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

new_df.to_csv('final_dataset.csv')

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
