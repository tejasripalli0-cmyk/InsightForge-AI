import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("InsightForge AI")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.write(df)

    st.subheader("Dataset Summary")
    st.write(df.describe())

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    st.write("Missing Values:", missing)
    st.write("Duplicate Rows:", duplicates)

    clean_df = df.drop_duplicates()

    for col in clean_df.select_dtypes(include='number').columns:
        clean_df[col].fillna(clean_df[col].mean(), inplace=True)

    st.subheader("Cleaned Dataset")
    st.write(clean_df)

    numeric_cols = clean_df.select_dtypes(include='number').columns

    if len(numeric_cols) > 0:

        fig, ax = plt.subplots()

        selected_col = st.selectbox(
        "Select column for visualization",
        numeric_cols
        )

        clean_df[selected_col].hist(ax=ax)

        st.pyplot(fig)

        st.subheader("AI Insights")

        st.write(
            f"""
            Dataset contains {len(clean_df)} records.
            Missing values were handled automatically.
            Duplicate records were removed.
            """
        )
