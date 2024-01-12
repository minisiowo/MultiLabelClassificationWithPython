import pandas as pd

# Wczytanie danych z pliku TSV
df = pd.read_csv("database/text_and_tags_100k.tsv", sep="\t")

# Liczenie ilości artykułów
number_of_articles = len(df)

# Obliczenie średniej długości artykułów
average_length = df["text"].apply(len).mean()

# Obliczenie średniej długości artykułów w słowach
average_word_length = df["text"].apply(lambda text: len(text.split())).mean()

print("Liczba artykułów: " + str(number_of_articles))
print("Średnia liczba słów w artykułach: " + str(average_word_length))
print("Średnia liczba znaków w artykułach: " + str(average_length))
