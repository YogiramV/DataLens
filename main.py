import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide")

st.title('DATALENS : Tool for analyzing datasets')

# Sidebar
file = st.sidebar.file_uploader(
    "Choose CSV file",
    type="csv"
)

# Tabs
info, eda = st.tabs(['Info', 'EDA'])

if file is not None:

    df = pd.read_csv(file)

    # =========================================
    # INFO TAB
    # =========================================

    with info:

        st.subheader("Uploaded CSV Data", divider=True)

        st.dataframe(df.head())

        # Dataset Info
        st.subheader('Dataframe Information', divider=True)

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        st.subheader('Columns')
        st.write(df.columns)

        st.subheader('Datatypes')
        st.write(df.dtypes)

        st.subheader('Missing Values')
        st.write(df.isna().sum())

        st.subheader('Correlation Heatmap(Only Numerical)')
        corr = df.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, width='stretch')

    # =========================================
    # EDA TAB
    # =========================================

    with eda:

        st.header('Summary Statistics', divider=True)
        st.write(df.describe(include='all'))

        st.header('Basic Plots', divider=True)

        st.subheader('Univariate Analysis', divider=True)
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected_col = st.selectbox(
            "Select Column", df.columns, index=None, placeholder="Select a column for analysis...")

        col1, col2 = st.columns(2)

        if selected_col in numeric_cols:
            # Histogram
            with col1:
                st.subheader('Histogram')
                fig1 = px.histogram(df, x=selected_col)
                st.plotly_chart(fig1, width='stretch')

            # Boxplot
            with col2:
                st.subheader('Boxplot')
                fig2 = px.box(df, y=selected_col)
                st.plotly_chart(fig2, width='stretch')
        elif selected_col is not None:
            # Bar Chart
            with col1:
                st.subheader('Bar Chart')
                counts = df[selected_col].value_counts()
                fig1 = px.bar(x=counts.index,
                              y=counts.values, labels={'x': selected_col, 'y': 'Count'})
                st.plotly_chart(fig1, width='stretch')
            # Pie Chart
            with col2:
                st.subheader('Pie Chart')
                counts = df[selected_col].value_counts()
                fig2 = px.pie(names=counts.index, values=counts.values)
                st.plotly_chart(fig2, width='stretch')

        st.subheader('Bivariate Analysis', divider=True)
        selected_col1 = st.selectbox(
            'Select column 1', df.columns, index=None, placeholder="Select a column for analysis...")
        selected_col2 = st.selectbox(
            'Select column 2', df.columns, index=None, placeholder="Select a column for analysis...")
        numvnum_chart = ['Scatter Plot', 'Line Plot', 'Heatmap']
        catvnum_chart = ['Box Plot', 'Violin Plot', 'Bar Plot']
        catvcat_chart = ["Grouped Bar", "Stacked Bar", "Heatmap"]

        if selected_col1 in numeric_cols and selected_col2 in numeric_cols:
            # Numerical Vs Numerical
            selected_chart = st.selectbox(
                'Select chart type', numvnum_chart, index=None, placeholder="Select required chart...",)
            if selected_chart == 'Scatter Plot':
                st.subheader('Scatter Plot')
                fig = px.scatter(df, x=selected_col1,
                                 y=selected_col2)
                st.plotly_chart(fig, width='stretch')
            elif selected_chart == 'Line Plot':
                st.subheader('Line Plot')
                fig = px.line(df, x=selected_col1, y=selected_col2)
                st.plotly_chart(fig, width='stretch')
            elif selected_chart == 'Heatmap':
                st.subheader('Heatmap')
                fig = px.density_heatmap(df, x=selected_col1, y=selected_col2)
                st.plotly_chart(fig, width='stretch')
        elif selected_col1 in numeric_cols or selected_col2 in numeric_cols:
            # Numerical Vs Categorical
            if selected_col1 in numeric_cols:
                num_col = selected_col1
                cat_col = selected_col2
            else:
                num_col = selected_col2
                cat_col = selected_col1

            selected_chart = st.selectbox(
                'Select chart type', catvnum_chart, index=None, placeholder="Select required chart...",)
            if selected_chart == 'Box Plot':
                st.subheader('Box Plot')
                fig = px.box(df, x=cat_col, y=num_col)
                st.plotly_chart(fig, width='stretch')
            elif selected_chart == 'Violin Plot':
                st.subheader('Violin Plot')
                fig = px.violin(df, x=cat_col, y=num_col)
                st.plotly_chart(fig, width='stretch')
            elif selected_chart == 'Bar Plot':
                st.subheader('Bar Plot')
                fig = px.bar(df, x=cat_col, y=num_col)
                st.plotly_chart(fig, width='stretch')
        elif selected_col1 is not None and selected_col2 is not None:
            # Categorical Vs Categorical
            selected_chart = st.selectbox(
                'Select chart type', catvcat_chart, index=None, placeholder="Select required chart...",)
            if selected_chart == 'Grouped Bar':
                st.subheader('Grouped Bar')
                fig = px.histogram(df, x=selected_col1,
                                   color=selected_col2, barmode='group')
                st.plotly_chart(fig, width='stretch')
            elif selected_chart == 'Stacked Bar':
                st.subheader('Stacked Bar')
                fig = px.histogram(df, x=selected_col1,
                                   color=selected_col2, barmode='stack')
                st.plotly_chart(fig, width='stretch')
            elif selected_chart == 'Heatmap':
                st.subheader('Heatmap')
                ct = pd.crosstab(df[selected_col1], df[selected_col2])
                fig = px.imshow(ct, text_auto=True,
                                color_continuous_scale='Blues')
                st.plotly_chart(fig, width='stretch')
