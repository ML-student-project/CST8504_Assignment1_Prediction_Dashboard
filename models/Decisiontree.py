import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df_train = pd.read_csv(r"C:\Users\graem\OneDrive\Documents\Algonquin\Fall Semester\CST8504 - AI Techniques\dir\train.csv")
df_test = pd.read_csv(r"C:\Users\graem\OneDrive\Documents\Algonquin\Fall Semester\CST8504 - AI Techniques\dir\test.csv")
df_validation = pd.read_csv(r"C:\Users\graem\OneDrive\Documents\Algonquin\Fall Semester\CST8504 - AI Techniques\dir\validation.csv")

X_train = df_train.drop(columns=['TenYearCHD'])
y_train = df_train['TenYearCHD']

X_test = df_test.drop(columns=['TenYearCHD'])
y_test = df_test['TenYearCHD']

X_validation = df_validation.drop(columns=['TenYearCHD'])
y_validation = df_validation['TenYearCHD']

model = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=3, min_samples_leaf=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))   # precision, recall, F1, accuracy
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

