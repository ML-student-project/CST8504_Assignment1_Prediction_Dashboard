import data_processor as dp
import model_trainer as mt
import predictor as p
import pandas as pd

def create_user_data():
    user_data = pd.DataFrame({
        "male": [1],
        "age": [34],
        "education": [2],
        "currentSmoker": [1],
        "cigsPerDay": [1],
        "BPMeds": [1],
        "prevalentStroke": [1],
        "prevalentHyp": [1],
        "diabetes": [1],
        "totChol": [130],
        "sysBP": [120],
        "diaBP": [12],
        "BMI": [12],
        "heartRate": [110],
        "glucose": [0.2],
        "TenYearCHD" : [-1]        #-- Will be removed by our preprocessor. But, expecting column as input.
    })
    return user_data

dp.main1()   #-- working
mt.main()

user_data = create_user_data()
print(user_data.shape)
prediction = p.dnn_predict(user_data)
print("DNN prediction is: " )
print(prediction[0])

prediction = p.decision_tree_predict(user_data)
print("DT prediction is: " )
print(prediction[0])
#
prediction = p.random_forest_predict(user_data)
print("RF  prediction is: " )
print(prediction[0])