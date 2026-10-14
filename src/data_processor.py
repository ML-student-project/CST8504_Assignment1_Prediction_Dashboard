# Load CSV from local file path or URL
from typing import AsyncGenerator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from streamlit.type_util import async_generator_to_sync

def load_data(filename = "./data/raw/framingham_heart_study.csv"):
    df = pd.read_csv(filename,na_values=['NA'])
    return df

# Histogram chart for all number columns.
def plot_data(df: pd.DataFrame):
    df.hist(bins=10, figsize=(10, 8))
    plt.tight_layout()
    plt.show()

##############################################################################################################
#                                                                                                            #
# Below is the code to clean and prepare our data so that it can be used for our decision tree and NN models #
#                                                                                                            #
##############################################################################################################

# Remove any duplicates in dataset
df.drop_duplicates()

# Check how many missing values for each variable
df.isna().sum()

# Set missing values of number of cigarettes to averge among smokers
smoker_avg = df.loc[df['currentSmoker'] == 1, 'cigsPerDay'].mean()
df.loc[(df['currentSmoker'] == 1) & (df['cigsPerDay'].isna()), 'cigsPerDay'] = smoker_avg

# Set missing glucose levels to average based on if they have diabetes or not
diabetes_G_avg = df.loc[df['diabetes'] == 1, 'glucose'].mean()
Non_diabetes_G_avg = df.loc[df['diabetes'] == 0, 'glucose'].mean()

df.loc[(df['diabetes'] == 1) & (df['glucose'].isna()), 'glucose'] = diabetes_G_avg
df.loc[(df['diabetes'] == 0) & (df['glucose'].isna()), 'glucose'] = Non_diabetes_G_avg

# create bins for age
age_edges = [0, 40, 50, 60, 1000]
age_labels = ['>= 40', '40-50', '50-60', '60+']

df['age_Group'] = pd.cut(
    df['age'],
    bins=age_edges,
    labels=age_labels,
    include_lowest=True
)

# create bins for cigarettes smoked per day
cigsPerDay_edges = [0, 1, 6, 16, 26, 1000]
cigsPerDay_labels = ['Zero', '1-5', '6-15', '16-25', '26+']

df['cigsPerDay_Group'] = pd.cut(
    df['cigsPerDay'],
    bins=cigsPerDay_edges,
    labels=cigsPerDay_labels,
    include_lowest=True
)

# create bins for total colestoral
totChol_edges = [0, 200, 240, 300, 1000]
totChol_labels = ['Normal', 'Elevated', 'High', 'Very High']

df['totChol_Group'] = pd.cut(
    df['totChol'],
    bins=totChol_edges,
    labels=totChol_labels,
    include_lowest=True
)

# create bins for system Blood pressure
sysBP_edges = [0, 120, 130, 140, 180, 300]
sysBP_labels = ['Normal', 'Elevated', 'Stage 1 High',
                'Stage 2 High', 'Severe']

df['sysBP_Group'] = pd.cut(
    df['sysBP'],
    bins=sysBP_edges,
    labels=sysBP_labels,
    include_lowest=True
)

# create bins for dia blood pressure
diaBP_edges = [0, 80, 90, 120, 200]
diaBP_labels = ['Normal', 'Stage 1 High',
                'Stage 2 High', 'Severe']

df['diaBP_Group'] = pd.cut(
    df['diaBP'],
    bins=diaBP_edges,
    labels=diaBP_labels,
    include_lowest=True
)

# create bins for BMI
BMI_edges = [0, 18.5, 25, 30, 35, 40, 100]
BMI_labels = ['Underweight', 'Normal', 'Overweight',
              'Obesity Class I', 'Obesity Class II',
              'Obesity Class III']

df['BMI_Group'] = pd.cut(
    df['BMI'],
    bins=BMI_edges,
    labels=BMI_labels,
    include_lowest=True
)

# create bins for heart rate
heartRate_edges = [0, 60, 100, 120, 300]
heartRate_labels = ['Low', 'Normal', 'Elevated', 'Very High']

df['heartRate_Group'] = pd.cut(
    df['heartRate'],
    bins=heartRate_edges,
    labels=heartRate_labels,
    include_lowest=True
)

# create bins for glucose levels
glucose_edges = [0, 70, 100, 126, 200, 1000]
glucose_labels = ['Low', 'Normal', 'Prediabetic',
                  'Diabetic Range', 'Very High']

df['glucose_Group'] = pd.cut(
    df['glucose'],
    bins=glucose_edges,
    labels=glucose_labels,
    include_lowest=True
)

# replace all other missing values by categorizing them as Not Known
df['totChol_Group'] = df['totChol_Group'].cat.add_categories("Not Known")
df['BMI_Group'] = df['BMI_Group'].cat.add_categories("Not Known")
df['heartRate_Group'] = df['heartRate_Group'].cat.add_categories("Not Known")

# create new dataframe using only variables we need 
dfClean = df.iloc[:, [15,0,1,2,3,5,6,7,8,16,17,18,19,20,21,22,23]].copy()

# one-hot encoding for the purpose of decision tree and NN models
encoded_df = pd.get_dummies(dfClean, dtype=int)
encoded_df


# Create testing, training and validating dataset
from sklearn.model_selection import train_test_split

# Split into training (70%) and temporary (30%) groups
train, temp = train_test_split(
    encoded_df,
    test_size=0.30,
    random_state=42
)

# Split temporary into validation (15%) and testing (15%)
validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42
)

# Export data to three separate CSV files
test.to_csv('test.csv', index=True)
train.to_csv('train.csv', index=True)
validation.to_csv('validation.csv', index=True)
