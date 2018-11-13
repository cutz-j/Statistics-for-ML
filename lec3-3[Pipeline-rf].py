### pipeline - randomforest ###
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

pipeline = Pipeline([('clf', RandomForestClassifier(criterion='gini'))])
parameters = {'clf__n_estimators':(1000, 2000, 3000),
              'clf__max_depth':(100, 200, 300),
              'clf__min_samples_split':(2, 3),
              'clf__min_samples_leaf':(1, 2)}