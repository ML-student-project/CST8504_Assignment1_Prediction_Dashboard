# Some code copied from https://medium.com/@tony.aloysius.77/building-a-prediction-web-app-using-streamlit-5bb881e545a

##- Just an outline. Not confirmed to work. Tam

# Creating a function for prediction
#def CHD_prediction(input_data):
#    # changing the input_data to numpy array
#    input_data_as_numpy_array = np.asarray(input_data)
#
#    # reshape the array as we are predicting for one instance
#    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#    scaler = StandardScaler()
#
#    # standardize the input data
#     std_data = scaler.fit_transform(input_data_reshaped)

#    # Tam - Call our preprocessor step instead of above standardize.
#
#    # Replace 'your_model.joblib' with your actual model file path
#    loaded_model = joblib.load('your_model.joblib')

#    return loaded_model.predict(std_data)


