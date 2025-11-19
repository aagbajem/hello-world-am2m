import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
import certifi
import urllib.request
from io import StringIO

ssl_context = ssl.create_default_context(cafile=certifi.where())

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'

# Load CSV from URL with SSL context
with urllib.request.urlopen(url, context=ssl_context) as response:
    data = response.read().decode('utf-8')

df = pd.read_csv(StringIO(data))


# Scatter Plot: Petal
plt.figure()
plt.scatter(df['petal_length'], df['petal_width'])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Iris Dataset – Petal Length vs Petal Width')
plt.show()


# Scatter Plot: Sepal
plt.figure()
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset – Sepal Length vs Sepal Width')
plt.show()


# Pairplot
sns.pairplot(df, hue='species')
plt.show()
