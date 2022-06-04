# Análise de Sentimentos (Google play scraper Dataset)


Aqui você encontrará uma simples implementação de um modelo ML treinado para realizar **análise de sentimentos**, usando a biblioteca [Keras](https://keras.io/) e um connjunto de dados *"scraped"* a partir das avaliações de aplicativos do [Google Play](https://play.google.com/store/games).

No pasta `\Data` existem dois arquivos, `data_clean.xlsx` e `data.xlsx`, que correspondem a uma versão *raw* e *pré-processada* dos dados coletados. O banco de dados contém **63876 amostras de avaliações** (todas as amostras estão em *língua portuguesa*). 

O arquivo `scrape.py` contém o crawler utilizado para criar o banco de dados, e pode ser reaproveitado para outras aplicações. Neste folder, também pode ser encontrado o arquivo `model_maker.py` e `hype_tune.py`, que foram utilizados para criar e afinar o modelo desenvolvido.

A aplicação foi desenvolvida utilizando [Flask](https://flask.palletsprojects.com/en/2.1.x/) e [Dash](https://dash.plotly.com/).