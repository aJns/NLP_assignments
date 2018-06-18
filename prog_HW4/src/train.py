import numpy as np
import sklearn.model_selection

import utils

data_file = "../data/ds_train.tsv"

(X, y) = utils.read_data_file(data_file)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)

np.array(X_train)
