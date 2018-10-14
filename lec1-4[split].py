import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Chapter01/mtcars.csv")

nrows = data.shape[0]
train_size, test_size, tuning_size = int(0.5 * nrows), int(0.25 * nrows), int(0.25 * nrows)

train_data, tmp_test_data = train_test_split(data, train_size=train_size, random_state=77)
test_data, tune_data = train_test_split(tmp_test_data, train_size=test_size, random_state=77)

print(train_data)
print(test_data)
print(tune_data)
