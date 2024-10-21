from __future__ import absolute_import, division, print_function, unicode_literals
from keras.src.utils.module_utils import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

from tensorflow.feature_column import *

import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow_estimator.python.estimator import estimator_lib as tf_estimator
from tensorflow.python.ops.distributions.categorical import Categorical

# Load dataset
dftrain = pd.read_csv('train.csv') # training data
dfeval = pd.read_csv('eval.csv') # testing data
print(dftrain.head())
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                          'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

print(feature_columns)

def input_fn():
    return tf1.data.Dataset.from_tensor_slices((dict(dftrain), y_train)).batch(32)

def eval_input_fn():
    return tf1.data.Dataset.from_tensor_slices((dict(dfeval), y_eval)).batch(32)

def model_fn(features, labels, mode):
    logits = tf.keras.layers.DenseFeatures(feature_columns)(features)
    loss = tf.losses.mean_squared_error(labels, logits)
    optimizer = tf.train.AdamOptimizer(0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

estimator = tf_estimator.Estimator(model_fn=model_fn)
estimator.train(input_fn=input_fn)
estimator.evaluate(input_fn=eval_input_fn)



#def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
 # def input_function():
  #  ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
   # if shuffle:
    #  ds = ds.shuffle(1000)
   # ds = ds.batch(batch_size).repeat(num_epochs)
   # return ds
  #return input_function

#train_input_fn = make_input_fn(dftrain, y_train)
#eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
#linear_est = tf_estimator.LinearClassifier(feature_columns=feature_columns)
#linear_est.train(train_input_fn)  # train
#result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

#clear_output()  # clears console output
#print(result['accuracy'])


