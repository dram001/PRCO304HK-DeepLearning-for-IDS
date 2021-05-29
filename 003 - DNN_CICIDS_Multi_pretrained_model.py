######################################################################################
#use pretrained model

import tensorflow as tf

model = tf.keras.models.load_model("DNN_CICIDS2017_MultiClass_model.h5")

model.summary()


######################################################################################
# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Cicids-merged.csv')
X = dataset.iloc[:, 8:83].values
y = dataset.iloc[:, 83].values


#####test code#####
#X = dataset.iloc[10000:50000, 8:83].values
#y = dataset.iloc[10000:50000, 83].values
###################




# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from keras.utils import np_utils
y = np_utils.to_categorical(y)


######################################################################################
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

######################################################################################
#Normalization
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
#https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)


# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

######################################################################################

# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#cm

#The Performance
from sklearn.metrics import confusion_matrix
#preds = classifier.predict(test)
pred_lbls = np.argmax(y_pred, axis=1)
true_lbls = np.argmax(y_test, axis=1)

model.evaluate(y_pred, y_test)

cm = confusion_matrix(true_lbls, pred_lbls)
cm

import seaborn as sns
sns.heatmap(cm, annot=True)
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')

############################################################
import matplotlib.pyplot as plt 
import itertools

class_names = ['Normal', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(12, 10))
plot_confusion_matrix(cm, classes=class_names,
                      title='CICIDS2017 Binary Friday DDos')

plt.show()

###############################################################

#output
classifier.save("DNN_CICIDS2017_MultiClass_model.h5")

