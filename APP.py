# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sbs import SBS
from sklearn.datasets import load_iris


st.set_page_config(page_title="SBS Feature Selector", layout="wide")
st.title("Sequential Backward Selection (SBS) App")

st.write("""
Upload your CSV dataset, choose the target column, and select how many features
you want to retain. The SBS algorithm will select the best subset of features
based on model accuracy.
""")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def load_default_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# File upload or default dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully.")
else:
    st.info("No file uploaded. Using Iris dataset as default.")
    df = load_default_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

with st.sidebar:
    st.header("Configuration")
    target_col = st.selectbox("Select Target Column", options=df.columns)

    if target_col:
        feature_cols = df.drop(columns=[target_col]).columns.tolist()
        k_max = len(feature_cols)
        k = st.slider("Select number of features to keep", min_value=1, max_value=k_max, value=k_max // 2)

        if st.button("Run SBS"):
            with st.spinner("Running SBS analysis..."):
                X = df[feature_cols].values
                y = df[target_col].values

                model = KNeighborsClassifier(n_neighbors=5)
                sbs = SBS(estimator=model, k_features=k)
                sbs.fit(X, y)

                selected_indices = list(sbs.indices_)
                selected_features = [feature_cols[i] for i in selected_indices]

                reduced_df = df[selected_features + [target_col]]

                st.session_state['selected_features'] = selected_features
                st.session_state['score'] = sbs.k_score_
                st.session_state['result_df'] = reduced_df

# Display results
if 'selected_features' in st.session_state:
    st.subheader("Selected Features")
    for feat in st.session_state['selected_features']:
        st.markdown(f"- {feat}")

    st.metric("Model Accuracy", f"{st.session_state['score']:.4f}")

if 'result_df' in st.session_state:
    st.subheader("Reduced Dataset Preview")
    st.dataframe(st.session_state['result_df'].head())

    csv = convert_df_to_csv(st.session_state['result_df'])
    st.download_button("Download Result CSV", csv, file_name="sbs_selected_features.csv", mime="text/csv")
