import streamlit as st
import seaborn as sns
import plotly.express as px
import sys
sys.path.append('./src')
# streamlit page config
import data_processor as dp

st.set_page_config(
    page_title="Data Exploration",  # the page title shown in the browser tab
    layout="wide",  # page layout : use the entire screen
)

# load data
df = dp.load_data()

# get column and TenYearCHD names
attributes = df.columns[:-1].tolist()
tenYearCHD = df['TenYearCHD'].unique().tolist()


# create sidebar with filtering options
with st.sidebar:
    # add header
    st.header("Filters", divider=True)
    # dropdown to select attributes
    selected_attribute = st.selectbox("Attribute: ", attributes, index=0)
    # multiselect to select species
    selected_tenYearCHD = st.multiselect("Chronic Heart Disease (CHD): ", tenYearCHD, placeholder="Filter by Ten Year CHD")

    # handle filter selections
    if not selected_tenYearCHD:
        selected_tenYearCHD = tenYearCHD
    filtered_df = df[df['TenYearCHD'].isin(selected_tenYearCHD)]



# histogram
def create_histogram(df, attribute):
    fig = px.histogram(df, x=attribute,
                            color="TenYearCHD",
                            marginal="box",
                            barmode="overlay")
    fig.update_traces(marker=dict(line=dict(
                                        width=1,
                                        color="rgba(100,100,100,0.5)")
                                        ))
    fig.update_layout(title=f"Histogram of {attribute}", hovermode="x unified")
    return fig

# violin plot
def create_violin_plot(df, attribute, points='all'):
    fig = px.violin(df, x="TenYearCHD", y=attribute,
                            color="TenYearCHD",
                            box=True,
                            points=points)
    fig.update_layout(title=f"Violin Plot of {attribute}")
    return fig

# columns to add 2 plots side by side
col1, col2 = st.columns(2)
# Histogram
with col1:
    hist_fig = create_histogram(filtered_df, selected_attribute)
    st.plotly_chart(hist_fig, use_container_width=True)
# Violin Plot
with col2:
    show_points = st.checkbox("Show individual points", value=True)
    violin_fig = create_violin_plot(filtered_df, selected_attribute, points="all" if show_points else False)
    st.plotly_chart(violin_fig, use_container_width=True)
