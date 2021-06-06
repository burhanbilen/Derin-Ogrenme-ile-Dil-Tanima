import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import pickle
import re

warnings.filterwarnings("ignore")

df = pd.read_excel("TEXTS.xlsx", usecols=[0, 1])

diller = {0:'Afrikanca',1:'Almanca',2:'Arapça',3:'Arnavutça',4:'Azerbaycan Türkçesi',5:'Belarusça'
         ,6:'Bengalce',7:'Boşnakça',8:'Bulgarca',9:'Burmaca',10:'Danca',11:'Endonezce',12:'Estonyaca'
         ,13:'Farsça',14:'Felemenkçe',15:'Filipince',16:'Fince',17:'Fransızca',18:'Gürcüce',19:'Hintçe'
         ,20:'Hırvatça',21:'Japonca',22:'Korece',23:'Lehçe',24:'Letonca',25:'Litvanca',26:'Macarca'
         ,27:'Makedonca',28:'Malezya Dili',29:'Norveççe',30:'Osmanlıca',31:'Portekizce',32:'Romence'
         ,33:'Rusça',34:'Slovakça',35:'Slovence',36:'Sırpça',37:'Tatarca',38:'Tayca',39:'Türkmence'
         ,40:'Türkçe',41:'Ukraynaca',42:'Urduca',43:'Uygurca',44:'Vietnamca',45:'Yunanca',46:'Çekçe'
         ,47:'Çince',48:'İbranice',49:'İngilizce',50:'İspanyolca',51:'İsveççe',52:'İtalyanca'
         ,53:'İzlandaca'}

df = df.sample(frac = 1)

def temizle(liste):
    yorumlar_temiz = []
    for text in liste:
        x = str(text)
        x = text.lower()
        x = re.sub(r'\<a href', ' ', x)
        x = re.sub(r'&amp;', '', x)
        x = re.sub(r'<br />', ' ', x)
        x = re.sub(r"^\s+|\s+$", "", x)
        x = re.sub(r'[_".`|´\{€¨~;>£$^♥♦•○☺½"：，<₺};%()|+&=*%.,!?:#$@\[\]/]', ' ', x)
        x = re.sub(r'\'', ' ', x)
        x = re.sub('\s{2,}', ' ', x)
        x = re.sub(r'\s+[a-zA-Z]\s+', ' ', x)
        x = re.sub(r'\^[a-zA-Z]\s+', ' ', x)
        x = re.sub(r'\s+', ' ', x, flags=re.I)
        x = re.sub(r'^b\s+', '', x)
        x = re.sub(r'\W', ' ', str(x))
        x = x.split()
        x = ' '.join(x)
        yorumlar_temiz.append(x)
    return yorumlar_temiz

X = np.array(temizle(df[0]))
y = np.array(df[1])

le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1,1)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

num_words = 70000 #batch_size
token = Tokenizer(num_words=num_words)
token.fit_on_texts(X)
dizi = token.texts_to_sequences(X)
X = sequence.pad_sequences(dizi) # "pre"
kelimeler = token.word_index

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = Sequential()
model.add(Embedding(num_words, X.shape[1])) #shape = input_length | çıktı: (batch_size, input_length, output_dim).
model.add(LSTM(64, activation='relu', dropout=0.2, recurrent_dropout = 0.1))
model.add(Dense(y.shape[1], activation='softmax'))

#1e-03
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

filepath="{val_accuracy:.2f}.h5"
my_callbacks = ModelCheckpoint(filepath)

history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1, callbacks=my_callbacks)

loss, accuracy = model.evaluate(X_test, y_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

tahmin = model.predict(X_test)
sonuc = np.argmax(tahmin, axis=1)

file_name = "tokenizer.pickle"
pickle.dump(token, open(file_name, 'wb'))
