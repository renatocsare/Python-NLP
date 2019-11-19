import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import pos_tag
import string

TEXT = "It was not clear that minds were changed. Certainly they were not inside the room, and most likely not elsewhere on Capitol Hill, where Republicans and Democrats were locked into their positions long ago. Nor were there any immediate signs that the hearing penetrated the general public. While major television networks broke into regular programming to carry it live, there was little sense of a riveted country putting everything aside to watch à la Watergate."

# Sentence tokenization
sentences = sent_tokenize(TEXT)

print("------- Original Text : \n" + TEXT + "\n\n")
print("------- Sentences : " + str(len(sentences)) + "\n")

i = 0
for sentence in sentences:
    i = i + 1
    print("-" + str(i) + " : " + sentence)

# Word Tokenization
print("\n-------- Tokenizaion\n")
tokens = word_tokenize(sentence)
print("Tokens Counting : " + str(len(tokens)))
i=0
for token in tokens:
    i = i + 1
    print("-" + str(i) + " : " + token)

# Removing Stop Words
print("\n-------- Stop Words\n")
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(string.punctuation)
cleanWords = [w for w in tokens if not w in stop_words]
print("------- Clean Words Count : " + str(len(cleanWords)))

i=0
for word in cleanWords:
    i = i + 1
    print("-" + str(i) + " : " + word)

# POS Tag
print("\n-------- POS Tag\n")
taggedWords = pos_tag(cleanWords)
print(nltk.help.upenn_tagset())
for taggedWord in taggedWords:
    print("- " + str(taggedWord))

# Stemming
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

# Lematization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print("\n-------- Lematization\n")
for word in cleanWords:
    print("- " + word + " : " + lemmatizer.lemmatize(word, pos="v"))

# Named Entity Recognition NER
from nltk import ne_chunk, sent_tokenize, word_tokenize

sentences = sent_tokenize(TEXT)

for sentence in sentences:
    words = word_tokenize(sentence)
    tags = pos_tag(words)
    ner = ne_chunk(tags)
    ner.draw()

