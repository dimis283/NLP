
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from transformers import MarianMTModel, MarianTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import io

# Μετάφραση


def translate(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# Φόρτωση δεδομένων
data = []
with open("fra.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:  # Αγνοούμε ελλιπείς γραμμές
            data.append({"english": parts[0], "french": parts[1]})

# Μετατροπή σε DataFrame
df = pd.DataFrame(data)
print(df.head())
df["english_length"] = df["english"].apply(lambda x: len(x.split()))
df["french_length"] = df["french"].apply(lambda x: len(x.split()))
print(df[["english_length", "french_length"]].mean())

english_words = " ".join(df["english"]).lower().split()
french_words = " ".join(df["french"]).lower().split()
print("Συχνότερες αγγλικές λέξεις:", Counter(english_words).most_common(10))
print("Συχνότερες γαλλικές λέξεις:", Counter(french_words).most_common(10))
#Bag of words
vectorizer = CountVectorizer()
X_english = vectorizer.fit_transform(df["english"])
X_french = vectorizer.fit_transform(df["french"])
#TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["english"])


# Φόρτωση μοντέλου (English -> French)
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
print(translate("Hello, how are you?"))  # Έξοδος: "Bonjour, comment allez-vous ?"


