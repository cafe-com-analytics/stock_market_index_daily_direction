from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from src.features.build_features import uniform_clustering, create_shifted_rt

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

    start_date = st.sidebar.date_input('Start date', start_date)
    end_date = st.sidebar.date_input('End date', end_date)

    df = yf.download(dct_market[page_selection]["index_name"], start=start_date, end=end_date)

    df["rt"] = (np.log(df["Close"]) - np.log(df["Close"].shift(periods=1)))

    df = create_shifted_rt(df, [1, 5, 37])

    df_clusterd = uniform_clustering(df[["Close", "rt", "rt-1", "rt-5", "rt-37"]], ["rt", "rt-1", "rt-5", "rt-37"])
    df_clusterd.dropna(how="any", axis=0, inplace=True)

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

    st.dataframe(df_clusterd)



if __name__ == '__main__':
    main()