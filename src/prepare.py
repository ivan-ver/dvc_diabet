import pandas as pd
import yaml
import sys
import random
import os


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split



if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)


params = yaml.safe_load(open("params.yaml"))["prepare"]
random.seed(params['seed'])

input_path = sys.argv[1]
os.makedirs("data/prepared")


def prepare_data(dataset):
    std_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    dataset['gender'] = dataset['gender'].apply(lambda x: 0 if x == 'Female' else 1)
    dataset = pd.get_dummies(dataset, columns=['gender', 'hypertension', 'heart_disease', 'smoking_history'], dtype=int)

    pipline_res = num_pipline = Pipeline([
        ('StandardScaler', StandardScaler()),
        ('MinMaxScaler', MinMaxScaler())
        ]).fit_transform(dataset[std_columns])
    for i, col_name in enumerate(std_columns):
        dataset[col_name] = pipline_res[:, i]
    return dataset



output_train = os.path.join("data", "prepared", "train.csv")
output_test = os.path.join("data", "prepared", "test.csv")

df = pd.read_csv(input_path)

test, train = train_test_split(df, train_size=params['split'], random_state=params['seed'], stratify=df['diabetes'])

test = prepare_data(test)
train = prepare_data(train)


train.to_csv(output_train, index=False)
test.to_csv(output_test, index=False)
