import numpy as np
import pandas as pd

df = pd.read_csv("../dataset/train.csv")

labels = df["evaluation"].values.astype(np.float32)

np.save("dataset/train_labels.npy", labels)