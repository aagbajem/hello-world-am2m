import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Load the data
df = pd.read_csv('train.csv')

# clean the data
# df.head()

# df.info()

# df

# select the features and the target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = ['Survived']

x = df[features]
y = df[target].values.ravel()

# fill missing values
x.loc[:, 'Age'] = x['Age'].fillna(x['Age'].median())
x.loc[:, 'Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])

# encode data
x = pd.get_dummies(x, drop_first=True)

# train and split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train the model   
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# predict the target
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)


print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: {conf_matrix}")
print(f"Classification Report: {class_report}")
print(f"ROC AUC: {roc_auc}")