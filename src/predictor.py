import pandas as pd
import numpy as np
import data_processor as dp
import joblib as jl


def dnn_predict (df):
    import tensorflow as tf

    # Load model and transformer separately
    loaded_model = tf.keras.models.load_model('./models/chd_dnn_model.keras')

    # Preprocess dataframe
    preprocessed_df = dp.pre_process_data(df)

    # Remove the TenYearCHD column from preprocessed data
    preprocessed_df.drop(columns=['TenYearCHD'], inplace=True)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(preprocessed_df)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Do prediction.
    pred_probs = loaded_model.predict(input_data_reshaped)

    return pred_probs

#----------------------------------------------------------------------------
def decision_tree_predict (df):

    import tensorflow as tf

    # Load model and transformer separately
    loaded_model = jl.load('./models/decision_tree.joblib')

    # Preprocess dataframe
    preprocessed_df = dp.pre_process_data(df)

    # Remove the TenYearCHD column from preprocessed data
    preprocessed_df.drop(columns=['TenYearCHD'], inplace=True)

    # Do prediction.
    pred_probs = loaded_model.predict(preprocessed_df)

    return pred_probs

def random_forest_predict (df):
    # Load model and transformer separately
    loaded_model = jl.load('./models/random_forest.joblib')

    # Preprocess dataframe
    preprocessed_df = dp.pre_process_data(df)

    # Remove the TenYearCHD column from preprocessed data
    preprocessed_df.drop(columns=['TenYearCHD'], inplace=True)

    # Do prediction.
    pred_probs = loaded_model.predict(preprocessed_df)

    return pred_probs