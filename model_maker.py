# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import io
import json 
import string
import unidecode
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from IPython.core.display import display, HTML
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten, Dense, LSTM
from keras import regularizers 

# ----------------------------------------------------------------------------------------#
# LOad Data
# ----------------------------------------------------------------------------------------#

data = pd.read_excel('Data\data_clean.xlsx')
data['content'] = data['content'].astype(str)
l = list(data['content'])
labelos = data['score']
labels = labelos.to_numpy()

# ----------------------------------------------------------------------------------------#
# Keras Tokenizer
# ----------------------------------------------------------------------------------------#

max_words = 47550   #Unique tokens in questions
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(l)
sequences = tokenizer.texts_to_sequences(l)
word_index = tokenizer.word_index
word_index["<PAD>"] = 47514  
print('Encontrados %s tokens únicos.' % len(word_index))

'''
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(sequences[0]), labels[0])
'''

padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
# ----------------------------------------------------------------------------------------#
# Split and Slice
# ----------------------------------------------------------------------------------------#

training_samples = 50000
test_samples = 13876

indices = np.arange(padded_sequences.shape[0])
np.random.shuffle(indices)
padded_sequences = padded_sequences[indices]
labels = labels[indices]
x_train = padded_sequences[:training_samples]
y_train = labels[:training_samples]
x_test = padded_sequences[training_samples: training_samples + test_samples]
y_test = labels[training_samples: training_samples + test_samples]

# ----------------------------------------------------------------------------------------#
# Keras Model
# ----------------------------------------------------------------------------------------#

model = keras.Sequential()
model.add(keras.layers.Embedding(max_words, 256, input_length=256)) 
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.8))
model.add(keras.layers.Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer= opt,
              loss='binary_crossentropy', 
              metrics=['accuracy'])

x_val = x_train[:25000]
partial_x_train = x_train[25000:]

y_val = y_train[:25000]
partial_y_train = y_train[25000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=5,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
print(model.evaluate(x_test,  y_test, verbose=1))
model.save('senti_model')

'''
# ----------------------------------------------------------------------------------------#
# Keras Model 2
# ----------------------------------------------------------------------------------------#

model = keras.Sequential()
model.add(keras.layers.Embedding(max_words, 256, input_length=256)) 
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

x_val = x_train[:25000]
partial_x_train = x_train[25000:]

y_val = y_train[:25000]
partial_y_train = y_train[25000:]

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
print(model.evaluate(x_test,  y_test, verbose=1))
model.save('senti_model_2')
'''
# ----------------------------------------------------------------------------------------#
# Keras Model Logs
# ----------------------------------------------------------------------------------------#

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig = go.Figure(layout={'template':'plotly_dark'})

fig.add_trace(go.Scatter(x=list(epochs), y=acc,
                         line_color='rgba(0, 102, 255, 0.5)', line=dict(width=3, dash='dash'), name='Acurácia (Treinamento)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Acurácia (Treinamento): %{y:.5f} acc <extra></extra>',
                         showlegend=True))
fig.add_trace(go.Scatter(x=list(epochs), y=val_acc,
                         line_color='rgba(255, 0, 0, 0.5)', line=dict(width=3, dash='dash'), name='Acurácia (Validação)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Acurácia (Validação): %{y:.2f} acc <extra></extra>',
                         showlegend=True))

fig.update_xaxes(showgrid=False, showline=False, mirror=False)
fig.update_yaxes(showgrid=True, ticksuffix=' acc')
fig.update_layout(
    paper_bgcolor='#242424',
    plot_bgcolor='#242424',
    hovermode='x unified',
    font_family='Open Sans',
    autosize=True,
    margin=dict(l=10, r=10, b=10, t=10),
    hoverlabel=dict(bgcolor='#242424', font_size=18, font_family='Open Sans')
)

fig.show()

fig2 = go.Figure(layout={'template':'plotly_dark'})

fig2.add_trace(go.Scatter(x=list(epochs), y=loss,
                         line_color='rgba(0, 102, 255, 0.5)', line=dict(width=3, dash='dash'), name='Loss (Treinamento)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Loss (Treinamento): %{y:.5f} loss <extra></extra>',
                         showlegend=True))
fig2.add_trace(go.Scatter(x=list(epochs), y=val_loss,
                         line_color='rgba(255, 0, 0, 0.5)', line=dict(width=3, dash='dash'), name='Loss (Validação)', mode='lines',
                         hoverlabel=dict(namelength=-1),
                         hovertemplate='Loss (Validação): %{y:.2f} loss <extra></extra>',
                         showlegend=True))

fig2.update_xaxes(showgrid=False, showline=False, mirror=False)
fig2.update_yaxes(showgrid=True, ticksuffix=' loss')
fig2.update_layout(
    paper_bgcolor='#242424',
    plot_bgcolor='#242424',
    hovermode='x unified',
    font_family='Open Sans',
    autosize=True,
    margin=dict(l=10, r=10, b=10, t=10),
    hoverlabel=dict(bgcolor='#242424', font_size=18, font_family='Open Sans')
)

fig2.show()

