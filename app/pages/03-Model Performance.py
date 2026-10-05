import streamlit as st
import seaborn as sns


#
# load data
df = sns.load_dataset('iris')
# get column and species names
attributes = df.columns[:-1].tolist()
species = df['species'].unique().tolist()

st.header("Predicated vs Actual Statistics")
st.dataframe(df.describe(), width='stretch')

st.header("Confusion Matrix")
st.dataframe(df.describe(), width='stretch')

