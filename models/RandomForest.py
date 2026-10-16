import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv(r"C:\Users\graem\OneDrive\Documents\Algonquin\Fall Semester\CST8504 - AI Techniques\dir\train.csv")
df_test = pd.read_csv(r"C:\Users\graem\OneDrive\Documents\Algonquin\Fall Semester\CST8504 - AI Techniques\dir\test.csv")
df_validation = pd.read_csv(r"C:\Users\graem\OneDrive\Documents\Algonquin\Fall Semester\CST8504 - AI Techniques\dir\validation.csv")

X_train = df_train.drop(columns=['TenYearCHD'])
y_train = df_train['TenYearCHD']

X_test = df_test.drop(columns=['TenYearCHD'])
y_test = df_test['TenYearCHD']

X_validation = df_validation.drop(columns=['TenYearCHD'])
y_validation = df_validation['TenYearCHD']

# Create the Random Forest
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=1
)

# Train the model
model.fit(X_train, y_train)

# Predict using validation data
y_val_pred = model.predict(X_validation)

print("Validation Accuracy:", accuracy_score(y_validation, y_val_pred))
print(classification_report(y_validation, y_val_pred))
print(confusion_matrix(y_validation, y_val_pred))

# Predict using test data
y_test_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))


