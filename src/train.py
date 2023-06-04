import pandas as pd
import numpy as np
import sys
import pickle
import yaml

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)

input = sys.argv[1]
output = sys.argv[2]

params = yaml.safe_load(open("params.yaml"))["train"]

train_df = pd.read_csv(input)

y = train_df['diabetes']
y = np.array(y).reshape(-1, 1)
X = train_df.drop('diabetes', axis=1)


model = BaggingClassifier(
    DecisionTreeClassifier(),
    bootstrap=True,
    n_estimators=500,
    n_jobs=-1,
    oob_score=True).fit(X, y)


with open(output, "wb") as fd:
    pickle.dump(model, fd)