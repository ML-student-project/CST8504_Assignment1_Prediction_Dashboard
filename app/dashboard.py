import streamlit as st
import seaborn as sns

# streamlit page config
st.set_page_config(
    page_title="Iris Dashboard",  # the page title shown in the browser tab
    layout="wide",  # page layout : use the entire screen
)

# load data
df = sns.load_dataset('iris')

# basic statistics
st.header("Summary Statistics")
st.dataframe(df.describe(), width='stretch')

st.header("Table Preview")
st.dataframe(df.head(), width='stretch')

st.header("Dataset Size")
st.write('\tNumber of rows:', df.shape[0])
st.write('\tNumber of columns:', df.shape[1])
