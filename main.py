from datetime import date, timedelta, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
import streamlit as st
import yfinance as yf

from src.features.build_features import uniform_clustering, create_shifted_rt
from src.models.train_model import train_model
from src.models.predict_model import predict_model

st.set_page_config(page_title="Market index direction w/ bayesian network", page_icon=":chart_with_upwards_trend:", layout='wide', initial_sidebar_state='auto')

def main():
    # Sidebar section:
    page_selection = st.sidebar.radio("Select a market:", ["Nikkey", "Bovespa"])

    dct_market = {
        "Nikkey":{"country":"Japan", "continent":"Asia", "index_name": "^N225"},
        "Bovespa":{"country":"Brazil", "continent":"America", "index_name": "^BVSP"}}

    st.markdown(f"# {page_selection}")

    end_date = date.today()
    start_date = end_date - timedelta(days=3150)

    # start_date = datetime.strptime('2004-11-02', '%Y-%m-%d')
    # end_date = datetime.strptime('2008-11-28', '%Y-%m-%d')

    start_date = st.sidebar.date_input('Start date', start_date)
    end_date = st.sidebar.date_input('End date', end_date)

    df = yf.download(dct_market[page_selection]["index_name"], start=start_date, end=end_date)

    df["rt"] = (np.log(df["Close"]) - np.log(df["Close"].shift(periods=1)))*100

    df = create_shifted_rt(df, [1, 5, 37])

    df_clustered = uniform_clustering(df[["Close", "rt", "rt-1", "rt-5", "rt-37"]], ["rt", "rt-1", "rt-5", "rt-37"])
    df_clustered.dropna(how="any", axis=0, inplace=True)

    lst_relations = [('cluster_rt-37', 'cluster_rt'), ('cluster_rt-5', 'cluster_rt'), ('cluster_rt-1', 'cluster_rt')]

    df_clustered = df_clustered[["rt", "cluster_rt-37", "cluster_rt-5", "cluster_rt-1", "cluster_rt"]]

    predict_n_days = 20

    model = train_model(df_clustered.iloc[:-predict_n_days], lst_relations)

    evidence = {
        'cluster_rt-37': df_clustered.iloc[-37]['cluster_rt'],
        'cluster_rt-5': df_clustered.iloc[-5]['cluster_rt'],
        'cluster_rt-1': df_clustered.iloc[-1]['cluster_rt']
        }

    predict = predict_model(model, evidence=evidence)

    st.text(f"Previsão para amanhã: {predict[0]}")

    resultado = {}

    for i in np.arange(1, predict_n_days+1):

        evidence = {
            'cluster_rt-37': df_clustered.iloc[-37-i]['cluster_rt'],
            'cluster_rt-5': df_clustered.iloc[-5-i]['cluster_rt'],
            'cluster_rt-1': df_clustered.iloc[-1-i]['cluster_rt']
            }
        
        predict = predict_model(model, evidence=evidence)

        resultado[i] = [predict[0]['cluster_rt'], df_clustered.iloc[i]['cluster_rt'], df_clustered.iloc[i]['rt']]

    resultado = pd.DataFrame.from_dict(resultado, orient='index')
    resultado.rename(columns={0: 'Previsão', 1: 'Real', 2:'rt'}, inplace=True)

    rt_mean = round(resultado.groupby(by=["Real"]).agg({"rt": ["min", "max","count", "mean"]}), 2)[("rt", "mean")]

    conditions = [
        resultado["Previsão"]==1.0, resultado["Previsão"]==2.0, resultado["Previsão"]==3.0
        , resultado["Previsão"]==4.0, resultado["Previsão"]==5.0, resultado["Previsão"]==6.0]

    choices = rt_mean.tolist()
    
    resultado["rt_predict"] = np.select(conditions, choices, default=np.nan)

    resultado = resultado[::-1]

    resultado["rt_predict_acumulado"] = resultado["rt_predict"].cumsum()
    resultado["rt_acumulado"] = resultado["rt"].cumsum()

    st.dataframe(resultado)

    rmse_uniform = mean_squared_error(resultado["rt"], resultado["rt_predict"], squared=False)

    acuracia = accuracy_score(resultado["Real"], resultado["Previsão"], normalize=True)

    st.text(f"Acurácia: {round(acuracia*100, 2)}%")
    st.text(f"RMSE: {round(rmse_uniform, 2)}%")
    
    # fig = plt.figure(figsize=(20, 4))
    # ax = fig.add_subplot(111)

    # ax.plot(df['Close'], label=dct_market[page_selection]["index_name"])
    
    # date_min = df.index.min()
    # date_max = df.index.max()
    # ax.xaxis.set_major_locator(plt.MaxNLocator(30))
    # ax.set_xlim(left=date_min, right=date_max)

    # ax.legend(loc='lower left', frameon=False)
    # plt.xticks(rotation=90)
    # st.pyplot(fig)

    st.line_chart(df[['Close']])

    st.line_chart(df["rt"])

    st.dataframe(df_clustered[['Close', 'rt', 'cluster_rt']].tail(10))

if __name__ == '__main__':
    main()