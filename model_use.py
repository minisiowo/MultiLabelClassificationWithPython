from joblib import load
import re  # importujemy bibliotekę do obsługi wyrażeń regularnych
import nltk  # importujemy bibliotekę do przetwarzania języka naturalnego

nltk.download("stopwords")  # pobieramy listę "stop words" z biblioteki nltk
from nltk.corpus import stopwords  # importujemy "stop words" z nltk.corpus
from nltk.stem.porter import PorterStemmer  # importujemy PorterStemmer do stemmingu

# wczytanie modelu
model = load("model_20k/model.joblib")
cv = load("model_20k/count_vectorizer.joblib")
mlb = load("model_20k/multilabel_binarizer.joblib")

# nowy tekst
with open("testing_article.txt", "r", encoding="utf-8") as file:
    new_text = file.read()

# oczyszczanie tekstu
article = re.sub(
    "[^a-zA-Z]", " ", new_text
)  # usuwamy wszystkie znaki, które nie są literami
article = article.lower()  # zamieniamy wszystkie litery na małe
article = article.split()  # dzielimy tekst na listę słów

ps = PorterStemmer()  # inicjalizujemy stemmer
article = [
    ps.stem(word) for word in article if not word in set(stopwords.words("english"))
]  # usuwamy "stop words" i stosujemy stemming
article = " ".join(article)  # łączymy słowa z powrotem w tekst

# przekształcanie tekstu w wektor cech
new_text_features = cv.transform([article]).toarray()

# uzycie modelu do przewidzenia tagów
new_text_pred = model.predict(new_text_features)

# przekształcenie przewidzianych etykiet binarnych na tagi
predicted_tags = mlb.inverse_transform(new_text_pred)
print(predicted_tags)

# print(mlb.classes_)
