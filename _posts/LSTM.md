

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
```


```python
train_df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
train_df.head()
```


```python
train_df, val_df = train_test_split(train_df, test_size = 0.2)
```


```python
embeddings_index = {}
f = open('/kaggle/working/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

print(f'Found {len(embeddings_index)} word vectors')
```


```python
def text_to_array(text):
    empty_emb =  np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empty_emb) for x in text]
    embeds+=[empty_emb] * (30 - len(embeds))
    return np.array(embeds)

val_vects = np.array([text_to_array(xtext) for xtext in tqdm(val_df['question_text'][:3000])])
val_y = np.array(val_df['target'][:3000])
```


```python
batch_size = 128

def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True:
        train_df = train_df.sample(frac = 1.)
        
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size,1]
            text_arr = np.array([text_to_array(text) for text in texts])
            
            yield text_arr, np.array(train_df['target'][i*batch_size:(i+1)*batch_size])
```


```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
```


```python
inp = Input(shape=(max_length,))
    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    predictions = Dense(1, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model
```


```python
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```


```python
mg = batch_gen(train_df)
model.fit_generator(mg, epochs = 20, steps_per_epoch = 1000, validation_data = (val_vects, val_y), verbose = True)
```


```python
batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())
```
