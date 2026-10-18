import data_processor as dp
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix ,  ConfusionMatrixDisplay
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib as jl

def train_decision_tree(X_train, y_train, X_test, y_test):

    model = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=3, min_samples_leaf=3)
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"DT Training finished, took {total_time:.2f} seconds")

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))  # precision, recall, F1, accuracy
    print(confusion_matrix(y_test, y_pred))

    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    # Plot the decision tree
    plt.figure(figsize=(50, 50))

    plot_tree(
        model,
        feature_names=X_train.columns,
        class_names=['Survived', 'Died'],
        filled=True,
        rounded=True,
        fontsize=15
    )

    plt.title("Decision Tree")
    plt.show()

    # Save the model
    jl.dump(model, './models/decision_tree.joblib')

def train_random_forest(X_train, y_train, X_test, y_test):
    import pandas as pd

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.ensemble import RandomForestClassifier

    # Create the Random Forest
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=1
    )

    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"RForest Training finished, took {total_time:.2f} seconds")

    # Predict using test data
    y_test_pred = model.predict(X_test)

    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))


    # Save the model
    jl.dump(model, './models/random_forest.joblib')



def train_dnn_model(x_train, y_train, x_val, y_val, x_test, y_test   ):
    start_time = time.time()

    model = tf.keras.Sequential([
        layers.Input(shape=(x_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        #layers.Dense(1)  # regression
        layers.Dense(1, activation="sigmoid")      # binary classification
        # layers.Dense(n_classes, activation="softmax")  # multi-class
    ])

    model.compile(optimizer="adam",
                  #loss="mse", metrics=["mae"])  # regression
                   loss="binary_crossentropy", metrics=["accuracy"] ) # classification

    early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model_batch_size = 32
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=200, batch_size=model_batch_size,
                        callbacks=[early_stop])

    end_time = time.time()
    total_time = end_time - start_time
    print(f"DNN Training finished, took {total_time:.2f} seconds")


    # Or evaluate final accuracy
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    def plot_metrics(history ) :
        # Assume 'history' is the object returned by model.fit(...)
        plot_type = 'val'
        # Plot Loss
        plt.figure()
        plt.plot(history.history['loss'], label='training Loss')
        plt.plot(history.history[plot_type + '_loss'], label= plot_type+' Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # Save the plot before showing it
        plt.savefig(f"./app/dnn_{plot_type}_loss.png", dpi=300, bbox_inches='tight')

    # Save the entire model
    model.save('./models/chd_dnn_model.keras')  # Default new format
    plot_metrics(history)


#=====================================================================================
def test_dnn_model (x_test, y_test):
    # Load model
    loaded_model = tf.keras.models.load_model('./models/chd_dnn_model.keras')

    # 1. Get raw predictions (probabilities)
    # model.predict() returns an array of shape (n_samples, n_classes)
    pred_probs = loaded_model.predict(x_test)

    # For binary: threshold at 0.5
    y_pred = (pred_probs > 0.5).astype(int)

    # Print confusion report.
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))  # precision, recall, F1, accuracy
    print(confusion_matrix(y_test, y_pred))

#----------------------------------------------------------------------------

def main():

    #-- Load data
    x_train = dp.load_data('./data/processed/x_train.csv')
    y_train = dp.load_data('./data/processed/y_train.csv')
    x_val = dp.load_data('./data/processed/x_val.csv')
    y_val = dp.load_data('./data/processed/y_val.csv')
    x_test = dp.load_data('./data/processed/x_test.csv')
    y_test = dp.load_data('./data/processed/y_test.csv')

    # remove header column from data for dnn
    dnn_x_train = df = x_train.iloc[1:]
    dnn_y_train = y_train.iloc[1:]
    dnn_x_val = val = x_val.iloc[1:]
    dnn_y_val = val = y_val.iloc[1:]
    dnn_x_test = x_test.iloc[1:]
    dnn_y_test = y_test.iloc[1:]

    train_dnn_model(dnn_x_train, dnn_y_train, dnn_x_val, dnn_y_val , dnn_x_test, dnn_y_test)
    test_dnn_model(x_test, y_test)

    train_decision_tree(x_train, y_train, x_test, y_test)
    train_random_forest(x_train, y_train, x_test, y_test)