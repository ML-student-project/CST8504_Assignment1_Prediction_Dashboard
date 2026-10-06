# Load CSV from local file path or URL
from typing import AsyncGenerator
import matplotlib.pyplot as plt
import pandas as pd
from streamlit.type_util import async_generator_to_sync

def load_data(filename = "./data/raw/framingham_heart_study.csv"):
    df = pd.read_csv(filename,na_values=['NA'])
    return df

# Histogram chart for all number columns.
def plot_data(df: pd.DataFrame):
    df.hist(bins=10, figsize=(10, 8))
    plt.tight_layout()
    plt.show()