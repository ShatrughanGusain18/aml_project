# -*- coding: utf-8 -*-
"""retraining.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UGSkEPePSu9zP6_YwGhSqX4HA-RwGaP0
"""

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd

import pickle

from google.colab import drive
drive.mount('/content/drive')

"""### transfer"""

# Read and divide the data
df_1 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/BIM/1/adv_samples_cluster1.csv")

df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/MIM/1/adload_adv_samples_cluster1.csv")
df_1 = pd.concat([df_1, df2], axis = 0)

df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/PGD/1/adv_samples_cluster1.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
ds_1 = df_1.values
df_1=df_1.drop('Unnamed: 0',axis=1)
print(df_1.head)

single_value = 'agent'
df_1['malwares'] = single_value

df_1.to_csv('/content/drive/MyDrive/research/Cleverhans/agent_adv.csv', index=False)

#Y1 = ds_1[:,-1]
#X_1 = ds_1[:,0:-1]
print(f"Dataset Size: {ds_1.shape[0]} Rows and {ds_1.shape[1]} Columns")

# Read and divide the data
#df_1 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/BIM/1/adv_samples_cluster1.csv")

df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/MIM/adv_samples_cluster2.csv")
df_1 = pd.concat([df_1, df2], axis = 0)

df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/PGD/adv_samples_cluster1.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
ds_1 = df_1.values
df_1=df_1.drop('Unnamed: 0',axis=1)
print(df_1.head)

single_value = 'adload'
df_1['malwares'] = single_value

df_1.to_csv('/content/drive/MyDrive/research/Cleverhans/adload_adv.csv', index=False)

#Y1 = ds_1[:,-1]
#X_1 = ds_1[:,0:-1]
print(f"Dataset Size: {ds_1.shape[0]} Rows and {ds_1.shape[1]} Columns")

# Read and divide the data
df_1 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/BIM/8/adv_samples_cluster1.csv")
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/BIM/8/adv_samples_cluster2.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/MIM/8/adload_adv_samples_cluster1.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/MIM/8/adv_samples_cluster2.csv")
df_1 = pd.concat([df_1, df2], axis = 0)

#df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/PGD/6/adv_samples_cluster1.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
ds_1 = df_1.values
df_1=df_1.drop('Unnamed: 0',axis=1)
print(df_1.head)

single_value = 'zeroaccess'
df_1['malwares'] = single_value

df_1.to_csv('/content/drive/MyDrive/research/Cleverhans/zeroaccess_adv.csv', index=False)

#Y1 = ds_1[:,-1]
#X_1 = ds_1[:,0:-1]
print(f"Dataset Size: {ds_1.shape[0]} Rows and {ds_1.shape[1]} Columns")

# Read and divide the data
df_1 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/BIM/3/adv_samples_cluster1.csv")
#df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/BIM/8/adv_samples_cluster2.csv")
#df_1 = pd.concat([df_1, df2], axis = 0)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/MIM/3/adload_adv_samples_cluster1.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/MIM/3/adv_samples_cluster2.csv")
df_1 = pd.concat([df_1, df2], axis = 0)

df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/PGD/3/adv_samples_cluster1.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
ds_1 = df_1.values
df_1=df_1.drop('Unnamed: 0',axis=1)
print(df_1.head)

single_value = 'obfuscator'
df_1['malwares'] = single_value

df_1.to_csv('/content/drive/MyDrive/research/Cleverhans/obfuscator_adv.csv', index=False)

#Y1 = ds_1[:,-1]
#X_1 = ds_1[:,0:-1]
print(f"Dataset Size: {ds_1.shape[0]} Rows and {ds_1.shape[1]} Columns")

# Read and divide the data
df_1 = pd.read_csv("/content/drive/MyDrive/research/dataset2.0/file_name.csv")
df_1=df_1.drop('malwares',axis=1)
df2 = pd.read_csv("/content/drive/MyDrive/research/dataset2.0/system_call.csv")
df_1 = pd.concat([df_1, df2], axis = 1)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/adload_adv.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/agent_adv.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/obfuscator_adv.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/vobfus_adv.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
df2 = pd.read_csv("/content/drive/MyDrive/research/Cleverhans/zeroaccess_adv.csv")
df_1 = pd.concat([df_1, df2], axis = 0)
ds_1 = df_1.values
Y1 = ds_1[:,-1]
X_1 = ds_1[:,0:-1]
print(f"Dataset Size: {ds_1.shape[0]} Rows and {ds_1.shape[1]} Columns")

df_1.to_csv('/content/drive/MyDrive/research/Cleverhans/adv_data.csv', index=False)

df_1.isnull().sum()

df_1 = df_1.fillna(df_1.mean())

pd.unique(Y1)

import re
df_1['malwares'] = df_1['malwares'].apply(lambda x: re.sub(r'\d+', '', x))

df_1['malwares']

y=df_1['malwares']

np.unique(y)

encoder = LabelEncoder()
encoded_Y1 = encoder.fit_transform(y)

print(encoded_Y1)

scaler_1 = MinMaxScaler()
scaler_1.fit(X_1)
X_1 = scaler_1.transform(X_1)

from sklearn.model_selection import train_test_split
X_train, x_rem_1, y_train, y_rem = train_test_split(X_1, encoded_Y1, train_size=0.8, random_state=10, stratify=encoded_Y1)
x_val_1, X_test, y_val, y_test = train_test_split(x_rem_1, y_rem, train_size=0.5, random_state=10, stratify=y_rem)

from keras.models import load_model
file_name = "/content/drive/MyDrive/research/dnnmodel.h5"
final_model = load_model(file_name)

from keras.models import load_model
model_1 = final_model

model_1.fit([X_train], y_train, epochs=150, batch_size = 8, verbose=1,  validation_data=([x_val_1], y_val))

scores = model_1.evaluate([X_test], y_test, verbose=0)
print("%s: %.2f%%" % (model_1.metrics_names[1], scores[1]*100))


predictions = model_1.predict([X_test])
predictions = np.argmax(predictions, axis=-1)
#y_test_labels = np.argmax(y_test, axis=1)
from sklearn.metrics import f1_score
#f1 = f1_score(y_test_labels, predictions, average='weighted')
#print('F1: ', "%.2f" % (f1*100))

# Confusion Matrix
cm = confusion_matrix(y_test_labels, predictions)   # makes a confusion matrix with no of samples predicted
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]   # convertes the values to percentage (cell/sum(row))
# print(cm)
# print(y_test_labels.shape)  # can check prediction size

# figure size, plot heatmap, change the values to percentage
plt.subplots(figsize=(10,10))
# heatmap = sns.heatmap(cm, annot=True, xticklabels = xlab, yticklabels= xlab, vmin=0, vmax=50, fmt='.0%', cmap='Blues')
heatmap = sns.heatmap(cm, annot=True, xticklabels = xlab, yticklabels= xlab, cmap='Blues')
# for cell in heatmap.texts: cell.set_text(cell.get_text() + " %")

def get_tpr_fnr_fpr_tnr(cm):
    dict_metric = dict()
    n = len(cm[0])
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    array_sum = sum(sum(cm))
    #initialize a blank nested dictionary
    for i in range(1, n+1):
        keys = str(i)
        dict_metric[keys] = {"TPR":0, "FNR":0, "FPR":0, "TNR":0}
    # calculate and store class-wise TPR, FNR, FPR, TNR
    for i in range(n):
        for j in range(n):
            if i == j:
                keys = str(i+1)
                tp = cm[i, j]
                fn = row_sums[i] - cm[i, j]
                dict_metric[keys]["TPR"] = tp / (tp + fn)
                dict_metric[keys]["FNR"] = fn / (tp + fn)
                fp = col_sums[i] - cm[i, j]
                tn = array_sum - tp - fn - fp
                dict_metric[keys]["FPR"] = fp / (fp + tn)
                dict_metric[keys]["TNR"] = tn / (fp + tn)
    return dict_metric




df = pd.DataFrame(get_tpr_fnr_fpr_tnr(cm)).transpose()
df = df.iloc[:,0:].apply(np.mean)*100
print(df)