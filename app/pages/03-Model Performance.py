import streamlit as st


# Define custom CSS to increase caption font size
st.markdown("""
<style>
    /* Target the caption element within st.image */
    div[data-testid="stImageCaption"] p {
        font-size: 30px; /* Adjust size as needed */
    }
</style>
""", unsafe_allow_html=True)

# Display a static image file
st.image("./app/Decision_tree-ConfusionMatrix.png", caption="Decision Tree Confusion Matrix")
