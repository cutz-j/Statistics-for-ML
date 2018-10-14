import pandas as pd
from sklearn.model_selection import train_test_split

original_data = pd.read_csv("Chapter01/mtcars.csv")
#print(orginal_data)

train_data, test_data = train_test_split(original_data, train_size=0.7, random_state=42)