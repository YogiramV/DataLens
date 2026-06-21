import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

st.title('DATALENS : Tool for analyzing datasets')

# Sidebar
file = st.sidebar.file_uploader(
    "Choose CSV file",
    type="csv"
)

# Tabs
info, eda, ml, cust = st.tabs(['Info', 'EDA', 'ML', 'Custom'])

if file is not None:

    df = pd.read_csv(file)

    # =========================================
    # INFO TAB
    # =========================================

    with info:

        st.subheader("Uploaded CSV Data", divider=True)

        st.dataframe(df)

        # Dataset Info
        st.subheader('Dataframe Information', divider=True)

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        col4, col5, col6 = st.columns(3)

        col4.subheader('Columns')
        col4.write(df.columns)

        col5.subheader('Datatypes')
        col5.write(df.dtypes)

        col6.subheader('Missing Values')
        col6.write(df.isna().sum())

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
                'Select chart type', numvnum_chart, index=None, placeholder="Select required chart...")
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

    # =========================================
    # ML TAB
    # =========================================

    with ml:
        df = df.dropna()
        target_col = st.selectbox(
            "Select the target column", df.columns, index=None, placeholder="Select required column...")

        if target_col and pd.api.types.is_any_real_numeric_dtype(df[target_col]):
            # Regression
            st.subheader('Regression')
            X = df[numeric_cols].drop(target_col, axis=1)
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=42)

            # Linear regression
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)

            # Decision Tree regression
            tree_reg = DecisionTreeRegressor()
            tree_reg.fit(X_train, y_train)

            # Random Forest regression
            rf_reg = RandomForestRegressor()
            rf_reg.fit(X_train, y_train)

            # Display predictions
            pred = pd.DataFrame({
                'Actual': y_test,
                'Predicted(Linear)': lin_reg.predict(X_test),
                'Predicted(Decision Tree)': tree_reg.predict(X_test),
                'Predicted(Random Forest)': rf_reg.predict(X_test)
            })
            st.dataframe(pred, hide_index=True)

            # Metrics
            models = {
                "Linear Regression": lin_reg,
                "Decision Tree": tree_reg,
                "Random Forest": rf_reg
            }

            results = []

            for name, model in models.items():
                y_pred = model.predict(X_test)

                results.append({
                    "Model": name,
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R2 Score": r2_score(y_test, y_pred)
                })

            df_results = pd.DataFrame(results)
            st.dataframe(df_results.style
                         .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='green')
                         .highlight_max(subset=['R2 Score'], color='green'), hide_index=True)

        elif target_col and not (pd.api.types.is_datetime64_any_dtype(df[target_col])):
            # Classification
            st.subheader('Classification')
            X = df.drop(target_col, axis=1)
            y = df[target_col]

            # Encode columns
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # Column segregation
            num_cols = X.select_dtypes(include='number').columns
            cat_cols = X.select_dtypes(['object', 'category']).columns

            # Transformers
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_cols),
                    ('cat', categorical_transformer, cat_cols)
                ]
            )

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # Decision Tree classifier
            tree_cl = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier())
            ])
            tree_cl.fit(X_train, y_train)
            tree_pred = tree_cl.predict(X_test)

            # Random Forest classifier
            rf_cl = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100,
                                                      random_state=42))
            ])
            rf_cl.fit(X_train, y_train)
            rf_pred = rf_cl.predict(X_test)

            pred = pd.DataFrame({
                'Actual': y_test,
                'Predicted(Decision Tree)': tree_pred,
                'Predicted(Random Forest)': rf_pred
            })
            pred['Actual'] = label_encoder.inverse_transform(pred['Actual'])
            pred['Predicted(Decision Tree)'] = label_encoder.inverse_transform(
                pred['Predicted(Decision Tree)'])
            pred['Predicted(Random Forest)'] = label_encoder.inverse_transform(
                pred['Predicted(Random Forest)'])
            st.dataframe(pred)

            # Metrics

            models = {
                'Decision Tree': tree_pred,
                'Random Forest': rf_pred
            }

            results = []

            for name, y_pred in models.items():

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(
                    y_test,
                    y_pred,
                    average='weighted'
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    average='weighted'
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    average='weighted'
                )

                # Store results
                results.append({
                    'Model': name,
                    'Accuracy': round(accuracy, 3),
                    'Precision': round(precision, 3),
                    'Recall': round(recall, 3),
                    'F1 Score': round(f1, 3)
                })
            results_df = pd.DataFrame(results)

            st.subheader("Model Comparison")

            st.dataframe(results_df.style.highlight_max(
                subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='green'), hide_index=True)

    # =========================================
    # CUSTOM TAB
    # =========================================
    model_dict = {'Linear regressor': [LinearRegression, ['tol', 'n_jobs']], 'Random forest regressor': [
        RandomForestRegressor, ['n_estimators', 'max_depth', 'max_leaf_nodes']]}

    def get_model(model_name):
        model_class, params = model_dict[model_name]
        return model_class(), params

    def model_cust(selected_model, col_name=None):
        if col_name == None:
            col_name = 'single'

        st.subheader('Parameter tuning', divider='blue')
        model, model_params = get_model(selected_model)
        st.write("Modifiable paramaters : ",
                 model_params)
        params = {}

        for i in model_params:
            params[i] = eval(st.text_input(i, value=0, key=col_name+str(i)))

        st.subheader('Model training', divider='green')
        y_selected = st.selectbox(
            'Choose data to predict', numeric_cols, index=None, placeholder="Select column", key=col_name+'y')
        x_selected = st.multiselect(
            'Choose data to fit', [
                x for x in numeric_cols if x != y_selected], placeholder="Select column", key=col_name+'x')
        st.write(model.set_params(**params))

        if x_selected and y_selected:
            X = df[x_selected]
            y = df[y_selected]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
            })
            st.write("Prediction for ", selected_model,
                     "with params : ", model.get_params())

            st.subheader('Prediction results')
            st.dataframe(pred, hide_index=True)

            metrics = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R² Score": r2_score(y_test, y_pred)
            }

            results_df = pd.DataFrame(
                [{"Metric": metric, "Value": round(value, 4)}
                    for metric, value in metrics.items()]
            )

            st.dataframe(results_df, use_container_width=True,
                         hide_index=True)

    with cust:
        comp = st.toggle('Compare models?')
        if not (comp):
            st.subheader('Model Customization')
            selected_model = st.selectbox(
                'Select model for customization', model_dict.keys(), index=None, placeholder="Select required model...")
            if selected_model:
                model_cust(selected_model)
        else:

            st.subheader('Compare models')

            m1 = st.selectbox(
                'Select model for customization', model_dict.keys(), index=None, placeholder="Select first model...")
            m2 = st.selectbox(
                'Select model for customization', model_dict.keys(), index=None, placeholder="Select second model...")

            if m1 and m2:

                model1, model2 = st.columns(
                    2, vertical_alignment='top', border=True)

                with model1:
                    model1.subheader('Model 1 : '+m1)
                    model_cust(m1, 'Left')

                with model2:
                    model2.subheader('Model 2 : '+m2)
                    model_cust(m2, 'Right')
