import streamlit as st
import seaborn as sns
import plotly.express as px

# streamlit page config
st.set_page_config(
    page_title="Data Exploration",  # the page title shown in the browser tab
    layout="wide",  # page layout : use the entire screen
)

# load data
df = sns.load_dataset('iris')
# get column and species names
attributes = df.columns[:-1].tolist()
species = df['species'].unique().tolist()


# create sidebar with filtering options
with st.sidebar:
    # add header
    st.header("Filters", divider=True)
    # dropdown to select attributes
    selected_attribute = st.selectbox("Attribute: ", attributes, index=0)
    # multiselect to select species
    selected_species = st.multiselect("Species: ", species, placeholder="Filter by species")

    # handle filter selections
    if not selected_species:
        selected_species = species
    filtered_df = df[df['species'].isin(selected_species)]

    filtered_df = df[df['species'].isin(selected_species)]


# pairwise scatter plot
def create_pairplot(df, attributes, species):
    fig = px.scatter_matrix(df, dimensions=attributes,
                                    color="species",
                                    opacity=0.6
                                   )
    fig.update_traces(diagonal_visible=False)
    return fig

# pairwise plot
st.subheader("Pairwise Scatter Plot (all attributes)")
pairplot_fig = create_pairplot(filtered_df, attributes, selected_species)
st.plotly_chart(pairplot_fig, use_container_width=True)

# histogram
def create_histogram(df, attribute):
    fig = px.histogram(df, x=attribute,
                            color="species",
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
    fig = px.violin(df, x="species", y=attribute,
                            color="species",
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
