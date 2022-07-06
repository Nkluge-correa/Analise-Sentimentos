# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import json 
import dash
import time
import string
import unidecode
import pandas as pd
import numpy as np
import tensorflow as tf
import dash_bootstrap_components as dbc
from tensorflow import keras
from keras import regularizers 
from dash import dcc, html, Output, Input, State
from dash.dependencies import Input, Output, State
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json


# ----------------------------------------------------------------------------------------#
# Load Model & Tokenizer
# ----------------------------------------------------------------------------------------#

model = keras.models.load_model('senti_model')

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    word_index = tokenizer.word_index

# ----------------------------------------------------------------------------------------#
# Dash Back-end
# ----------------------------------------------------------------------------------------#

def textbox(text, box='other'):
    style = {
        'max-width': '55%',
        'width': 'max-content',
        'padding': '10px 15px',
        'border-radius': '25px',
    }

    if box == 'self':
        style['margin-left'] = 'auto'
        style['margin-right'] = 0

        color = 'primary'
        inverse = True

    elif box == 'other':
        style['margin-left'] = 0
        style['margin-right'] = 'auto'

        color = 'light'
        inverse = False

    else:
        raise ValueError('Incorrect option for `box`.')

    return dbc.Card(text, style=style, body=True, color=color, inverse=inverse)

conversation = html.Div(
    style={
        'width': '80%',
        'max-width': '600px',
        'height': '35vh',
        'margin': 'auto',
        'margin-top': '100px',
        'overflow-y': 'auto',
    },
    id='display-prediction',
)


controls = dbc.InputGroup(
    style={'width': '80%', 'max-width': '600px', 'margin': 'auto'},
    children=[
        dbc.Input(id='user-input', placeholder='Escreva um comentário...', type='text'),
        dbc.InputGroup(dbc.Button('Enviar', size= 'lg', id='submit')),
    ],
)

# ----------------------------------------------------------------------------------------#
# Dash app
# ----------------------------------------------------------------------------------------#

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
app.title = 'Classificador de Sentimento'

# ----------------------------------------------------------------------------------------#
# Layout
# ----------------------------------------------------------------------------------------#

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1('Classificador de Sentimentos  ', style={'color':'#242020',
                                                    'font-style': 'bold', 
                                                    'margin-top': '15px',
                                                    'margin-left': '15px',
                                                    'display':'inline-block'}),
        html.H1('  🤖', style={'color':'#2a9fd6',
                            'font-style': 'bold', 
                            'margin-top': '15px',
                            'margin-left': '15px',
                            'display':'inline-block'}),
        html.Hr(),
        dbc.Row([
        dbc.Col([
            dcc.Store(id='store-data', data=''),
            dcc.Loading(id='loading_0', type='circle', children=[conversation]),
            controls,
        ], md = 12),
        ]),
    ],
)

# ----------------------------------------------------------------------------------------#
# Funções
# ----------------------------------------------------------------------------------------#

@app.callback(
    Output('display-prediction', 'children'), 
    [
        Input('store-data', 'data')]
)
def update_display(sentiment_analysis):
    time.sleep(2)
    return [
        textbox(sentiment_analysis, box='self') if i % 2 == 0 else textbox(sentiment_analysis, box='other')
        for i, sentiment_analysis in enumerate(sentiment_analysis)
    ]



@app.callback(
    [
     Output('store-data', 'data'),
     Output('user-input', 'value')
    ],

    [
     Input('submit', 'n_clicks'), 
     Input('user-input', 'n_submit')
    ],

    [
     State('user-input', 'value'), 
     State('store-data', 'data')
    ]
)

def run_senti_model(n_clicks, n_submit, user_input, sentiment_analysis):
    if n_clicks == 0:
        sentiment_analysis = []
        sentiment_analysis.append('🎭')
        sentiment_analysis.append('Como você se sente?')
        return sentiment_analysis, ''


    if user_input is None or user_input == '':
        sentiment_analysis = []
        sentiment_analysis.append('🎭')
        sentiment_analysis.append('Como você se sente?')
        return sentiment_analysis, ''

    else:
        texto = user_input
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.lower()
        texto = unidecode.unidecode(texto)
        texto = [texto]
        text = tokenizer.texts_to_sequences(texto)
        padded_text = keras.preprocessing.sequence.pad_sequences(text,
                                                                value=word_index["<PAD>"],
                                                                padding='post',
                                                                maxlen=256)
        
        prediction = model.predict(padded_text)

        if prediction[0][0] <= 0.5:
            pred = 1 - prediction[0][0]
            pred = '{:,.2f}'.format(pred * 100) + ' %'
            response = f'Sentimento Negativo 😔 \n {pred}'

        elif prediction[0][0] >= 0.5:
            pred = '{:,.2f}'.format(prediction[0][0] * 100) + ' %'
            response = f'Sentimento Positivo 😊 \n {pred}'
    
        sentiment_analysis = []
        sentiment_analysis.append(user_input)
        sentiment_analysis.append(response)

        return sentiment_analysis, ''

if __name__ == '__main__':
    app.run_server(debug=False)
