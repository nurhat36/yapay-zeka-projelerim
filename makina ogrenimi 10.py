import pandas as pd
import numpy as np
import re
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Hatalı satırları atla
dataframe = pd.read_csv("Restaurant_Reviews.csv", nrows=321)

# NaN değerleri kontrol et ve temizle
print(dataframe.isnull().sum())

# NaN değerleri olan satırları kaldır
dataframe = dataframe.dropna()

print(dataframe.head())

nltk.download("stopwords")
ps = PorterStemmer()
derlem = []

# Yorumdaki harfleri boşluk ile değiştirme
for i in range(len(dataframe)):
    yorum = re.sub('[^a-zA-Z]', " ", dataframe["Review"].iloc[i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum)
    derlem.append(yorum)

print(derlem)

cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()
Y = dataframe.iloc[:, 1].values

print(X)
print(Y)

# Veriyi böl
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

print(x_train)
print("y")
print(y_train)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
