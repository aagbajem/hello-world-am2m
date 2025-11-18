import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('train.csv')

# Select the features and the target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

x = df[features]
y = df['Survived']

# Fill missing values
x.loc[:, 'Age'] = x['Age'].fillna(x['Age'].median())
x.loc[:, 'Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])
x.loc[:, 'Fare'] = x['Fare'].fillna(x['Fare'].median())

# Encode categorical data
x = pd.get_dummies(x, drop_first=True)

# Train and split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the data (optional for Random Forest, but can help)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples in leaf node
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

rf_model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(x_test_scaled)
y_prob = rf_model.predict_proba(x_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(class_report)
print("\nConfusion Matrix:")
print(conf_matrix)