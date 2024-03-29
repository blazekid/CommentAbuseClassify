import numpy as np 
import pandas as pd 
import cPickle as pickle
import h5py
import tweepy

import keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Activation
from keras.layers import PReLU
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Flatten

from keras.layers.normalization import BatchNormalization
from keras.models import load_model

max_features = 25000
maxlen = 200
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = maxlen

consumer_key ="oZqUgY0vLZWpD25OAFhm7uLwL"
consumer_secret ="9sgHQc3wbonE5bS0sK6tGFHvAZns0LP7aBN1JOaVjHLRRK65MY"
access_token ="995775944987062272-VzLQTuudo6qcnc6wMxFbARVS39hYWd6"
access_token_secret ="gf7I1X2bn509GJ7lTGThjhHw66hYybflW1pCTwrA4iipY"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

print('Start Tokenizing')
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
train = train.sample(frac=1)



list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
#list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
#X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


def get_model():
    inp = Input(shape=(maxlen, ))
    x_3 = Embedding(max_features, EMBEDDING_DIM)(inp)
    cnn1 = Conv1D(128, 2, padding='same', strides=1, activation='relu')(x_3)
    cnn2 = Conv1D(128, 3, padding='same', strides=1, activation='relu')(x_3)
    cnn = keras.layers.concatenate([cnn1, cnn2], axis=-1)
    cnn1 = Conv1D(64, 2, padding='same', strides=1, activation='relu')(cnn)
    cnn1 = MaxPooling1D(pool_size=100)(cnn1)
    cnn2 = Conv1D(64, 3, padding='same', strides=1, activation='relu')(cnn)
    cnn2 = MaxPooling1D(pool_size=100)(cnn2)
    cnn = keras.layers.concatenate([cnn1, cnn2], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    x = Dense(400, kernel_initializer='he_normal')(drop)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sgd = keras.optimizers.SGD(lr=0.001)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy', 'binary_crossentropy'])
    return model



batch_size = 128
epochs = 20
k = 2
result = []



print('Start KFold')
from sklearn.model_selection import KFold
kf = KFold(n_splits=k, shuffle=False)
for train_index, test_index in kf.split(X_t):
    X_Train = X_t[train_index]
    Y_Train = y[train_index]
    X_Test = X_t[test_index]
    Y_Test = y[test_index]
    file_path = "weights_base_[KFold_CNN].best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_binary_crossentropy", mode="min", patience=8)
    callbacks_list = [checkpoint, early]
    model = get_model()
    print("got model!!!")
    model.fit(X_Train, Y_Train, batch_size=batch_size, epochs=epochs, validation_data=(X_Test, Y_Test), verbose=1)
    print("model fitted")
    model.save('my_newmodelcnn.h5')
    # model.load_weights(file_path)
    # result.append(model.predict(X_te))
    exit() 

ans = 'Y'

model = load_model('my_newmodelcnn.h5')
while (ans=='Y'):
    testInput = raw_input('Enter a sentence for abuse classification : ')
    listx = []
    listx.append(testInput)
    list_tokenized_test_x = tokenizer.texts_to_sequences(listx)
    X_te_x = sequence.pad_sequences(list_tokenized_test_x, maxlen=maxlen) 
    fin_res = model.predict(X_te_x)
    for x in range(0,6):
        print ( list_classes[x]+" = " + str(fin_res[0][x])) 
    if(fin_res[0][0] <= 0.50):
        api.update_status(status = testInput)
        print ("Your comment have been Approved and Posted!!!")
    else:
        print ("Your comment is abusive. Please refrain from using such language!")
    ans = raw_input('Want to enter more sentence? Y/N');
    


