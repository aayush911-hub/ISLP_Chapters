import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

import statsmodels.api as sm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize)

from ISLP import confusion_table
from ISLP.models import contrast

from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA,
     QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Smarket= load_data('Smarket')


from IPython.core.magic import register_cell_magic
import os
@register_cell_magic
def save(line, cell):
    parent_path= os.path.join(os.path.dirname(__file__), "Imports.py")
    with open(parent_path, 'a') as f:
        f.write("\n" + cell + "\n")
    print(f"Successfully added the cell code in the file {parent_path}")

train= Smarket.Year<2005

X= Smarket[['Lag1', 'Lag2']]
y= Smarket.Direction

X_train, X_test= X.loc[train], X.loc[~train]
y_train, y_test= y.loc[train], y.loc[~train]


