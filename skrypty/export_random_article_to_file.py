import pandas as pd

# Wczytaj dane z pliku TSV
df = pd.read_csv("database/text_and_tags_20k.tsv", sep="\t")

# Wybierz losowy artykuł
random_article = df.sample()["text"].values[0]

# Zapisz artykuł do pliku
with open("testing_article.txt", "w", encoding="utf-8") as file:
    file.write(random_article)
