import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#The Data
with open('kddcup.names', 'r') as infile:
    kdd_names = infile.readlines()
kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]
kdd_cols += ['class', 'difficulty']

kdd = pd.read_csv('KDDTrain+.txt', names=kdd_cols)
kdd_t = pd.read_csv('KDDTest+.txt', names=kdd_cols)

kdd.head()

kdd_cols = [kdd.columns[0]] + sorted(list(set(kdd.protocol_type.values))) + sorted(list(set(kdd.service.values))) + sorted(list(set(kdd.flag.values))) + kdd.columns[4:].tolist()

attack_map = [x.strip().split() for x in open('training_attack_types', 'r')]
attack_map = {k:v for (k,v) in attack_map}

attack_map

kdd['class'] = kdd['class'].replace(attack_map)
kdd_t['class'] = kdd_t['class'].replace(attack_map)

def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)


def log_trns(df, col):
    return df[col].apply(np.log1p)

cat_lst = ['protocol_type', 'service', 'flag']
for col in cat_lst:
    kdd = cat_encode(kdd, col)
    kdd_t = cat_encode(kdd_t, col)
    
log_lst = ['duration', 'src_bytes', 'dst_bytes']
for col in log_lst:
    kdd[col] = log_trns(kdd, col)
    kdd_t[col] = log_trns(kdd_t, col)
    
kdd = kdd[kdd_cols]
for col in kdd_cols:
    if col not in kdd_t.columns:
        kdd_t[col] = 0
kdd_t = kdd_t[kdd_cols]

kdd.head()


difficulty = kdd.pop('difficulty')
target = kdd.pop('class')
y_diff = kdd_t.pop('difficulty')
y_test = kdd_t.pop('class')

target = pd.get_dummies(target)
y_test = pd.get_dummies(y_test)

target

y_test

target = target.values
train = kdd.values
test = kdd_t.values
y_test = y_test.values

min_max_scaler = MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)

train.shape

for idx, col in enumerate(list(kdd.columns)):
    print(idx, col)



#The Model

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout
from keras.layers.embeddings import Embedding

def build_network():

    models = []
    model = Sequential()
    model.add(Dense(64, input_dim=122))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

NN = build_network()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')


NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=128, callbacks=[early_stopping])



#The Performance
from sklearn.metrics import confusion_matrix
preds = NN.predict(test)
pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test, axis=1)

NN.evaluate(test, y_test)

confusion_matrix(true_lbls, pred_lbls)


from sklearn.metrics import f1_score
f1_score(true_lbls, pred_lbls, average='weighted')

#output
NN.save("my_h5_NSLKDD_model.h5")

#new pre






