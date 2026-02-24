import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Amazon Review Dashboard",
                   layout="wide")

st.title("Amazon Review Sentiment Analysis Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("D:/downloads/Reviews.csv/Reviews.csv")
    df = df.head(500)   
    return df

df = load_data()

col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", len(df))
col2.metric("Average Rating", round(df["Score"].mean(),2))
col3.metric("Unique Users", df["UserId"].nunique())

st.divider()

def sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Score"].apply(sentiment)

st.subheader("Sentiment Distribution")

fig1 = px.pie(
    df,
    names="Sentiment",
    title="Customer Sentiment"
)

st.plotly_chart(fig1, use_container_width=True)

st.subheader("Rating Analysis")

fig2 = px.histogram(
    df,
    x="Score",
    nbins=5,
    title="Ratings Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Buy Again Prediction")

df["Buy_Again"] = df["Score"].apply(
    lambda x: "Yes" if x >= 4 else "No"
)

buy_counts = df["Buy_Again"].value_counts().reset_index()
buy_counts.columns = ["Buy Again", "Count"]

fig3 = px.bar(
    buy_counts,
    x="Buy Again",
    y="Count",
    title="Customer Repurchase Prediction"
)

st.plotly_chart(fig3, use_container_width=True)

st.subheader("Review Explorer")

selected_sentiment = st.selectbox(
    "Filter by Sentiment",
    df["Sentiment"].unique()
)

filtered = df[df["Sentiment"] == selected_sentiment]

st.dataframe(filtered[["Score", "Summary", "Text"]].head(10))
