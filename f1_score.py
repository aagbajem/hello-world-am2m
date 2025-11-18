import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score

df = pd.read_csv('train.csv')

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

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train, y_train)
y_pred_lr = log_reg.predict(x_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

f1_lr = f1_score(y_test, y_pred_lr)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Logistic Regression F1-Score: {f1_lr:.4f}")
print(f"Random Forest F1-Score: {f1_rf:.4f}")