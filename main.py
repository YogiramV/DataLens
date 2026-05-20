import streamlit as st
import pandas as pd
import os

st.title('DATALENS : tool for analyzing datasets')

file = st.file_uploader("Choose csv file", type="csv")

if file is not None:
    df=pd.read_csv(file)
    st.subheader("Uploaded CSV Data")
    st.dataframe(df)
    
    st.subheader('Dataframe information')
    st.text('Shape')
    st.write(df.shape)
    st.text('Columns')
    st.write(df.columns)
    st.text('Datatypes')
    st.write(df.dtypes)
    st.text('Missing values')
    st.write(df.isna().sum())

