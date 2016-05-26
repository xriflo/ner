import nltk
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def nltk_test():
    doc = 'John lives in London and works at IBM. Lucy works at Intel in Berlin. ' \
          'Catalin works at Integrisoft in Bucharest'
    sentences = nltk.sent_tokenize(doc)
    print "Sentences found: "
    print sentences

    print "\n"
    words = [nltk.word_tokenize(sent) for sent in sentences]

    print "Words found: "
    print words
    print "\n"

    word_tags = [nltk.pos_tag(word) for word in words]

    print "Word tags found: "
    print word_tags
    print "\n"


def generate_sentences(corpus_file):
    document = open(corpus_file)
    lines = [line.rstrip("\n") for line in document]

    last_index = 0
    sentences = []

    for i in range(len(lines)):
        if lines[i] == '':
            sentences.append(lines[last_index:i])
            last_index = i + 1

    sentences = [[(word.split(' ')[0], word.split(' ')[1:]) for word in sent] for sent in sentences]
    return sentences


def generate_features(sentence):
    features = []

    for i, (word, tag_tuple) in enumerate(sentence):
        word_features = [word, tag_tuple[0]]

        previous2PosTag = "None"
        previousPosTag = "None"

        previous2Word = "None"
        previousWord = "None"

        nextPosTag = "None"
        next2PosTag = "None"

        nextWord = "None"
        next2Word = "None"

        phraseStart = i == 0
        phraseEnd = i == len(sentence) - 1
        if i == 0 and len(sentence) > 2:
            nextPosTag = sentence[i + 1][1][0]
            next2PosTag = sentence[i + 2][1][0]

            nextWord = sentence[i + 1][0]
            next2Word = sentence[i + 2][0]
        if i == 0 and len(sentence) > 1:
            nextPosTag = sentence[i + 1][1][0]
            nextWord = sentence[i + 1][0]
        elif i == len(sentence) - 2:
            previousPosTag = sentence[i - 1][1][0]
            previous2PosTag = sentence[i - 2][1][0]

            previousWord = sentence[i - 1][0]
            previous2Word = sentence[i - 2][0]
        elif i == len(sentence) - 1:
            previousPosTag = sentence[i - 1][1][0]
            previousWord = sentence[i - 1][0]
        else:
            nextPosTag = sentence[i + 1][1][0]
            next2PosTag = sentence[i + 2][1][0]

            nextWord = sentence[i + 1][0]
            next2Word = sentence[i + 2][0]

            previousPosTag = sentence[i - 1][1][0]
            previous2PosTag = sentence[i - 2][1][0]

            previousWord = sentence[i - 1][0]
            previous2Word = sentence[i - 1][0]

        word_features.append(previousPosTag)
        word_features.append(previous2PosTag)

        word_features.append(previousWord)
        word_features.append(previous2Word)

        word_features.append(nextPosTag)
        word_features.append(next2PosTag)

        word_features.append(nextWord)
        word_features.append(next2Word)
        word_features.append(phraseStart)
        word_features.append(phraseEnd)

        #word_features.append(word[0].isupper())
        word_features.append(tag_tuple[2])
        features.append(word_features)
    return features


def generate_data(sentences, list_of_features):
    train_data = []
    for i in range(len(sentences)):
        features = generate_features(sentences[i])
        train_data = train_data + features

    return pd.DataFrame(train_data, columns=list_of_features)


def convert_data_to_numeric(dataframe, list_of_features):
    for i in range(len(list_of_features)):
        feature = list_of_features[i]
        feature_dict = {}
        feature_dict.setdefault(90000)

        unique_values_list = pd.unique(dataframe[feature].ravel())

        for j in range(0, len(unique_values_list)):
            feature_dict[unique_values_list[j]] = j

        dataframe[feature] = dataframe[feature].apply(lambda x: feature_dict.get(x))

    return dataframe


corpus_file_name = 'eng.train.txt'

list_of_predictors = ["Word",
                      "PosTag",
                      "PreviousPosTag",
                      "Previous2PosTag",
                      "PreviousWord",
                      "Previous2Word",
                      "NextPosTag",
                      "Next2PosTag",
                      "NextWord",
                      "Next2Word",
                      "PhraseStart",
                      "PhraseEnd"
                      ]

list_of_features = list_of_predictors + ["NamedEntity"]


print list_of_features
list_of_sentences = generate_sentences(corpus_file_name)

test_set = list_of_sentences[10000:]
train_set = list_of_sentences[1:10000]

print ("Test set length : %d" % len(test_set))
print ("Train set length : %d" % len(train_set))
print ("\n")

train_data_frame = generate_data(train_set, list_of_features)
train_data_frame.to_csv("NERC_train.csv", index=False)
numeric_train_frame = convert_data_to_numeric(train_data_frame, list_of_features)
numeric_train_frame.to_csv("NERC_numeric_train.csv", index=False)

test_data_frame = generate_data(test_set, list_of_features)
test_data_frame.to_csv("NERC_test.csv", index=False)

NamedEntityValues = test_data_frame['NamedEntity']

new_test_data_frame = test_data_frame.loc[test_data_frame["NamedEntity"].isin(["I-ORG", "I-PER", "I-LOC"])]
# print new_test_data_frame[["Word"]]

numeric_test_frame = convert_data_to_numeric(test_data_frame, list_of_features)
new_numeric_test_frame = convert_data_to_numeric(new_test_data_frame, list_of_features)
numeric_test_frame.to_csv("NERC_numeric_test.csv", index=False)

# numeric_test_frame = df = numeric_test_frame.drop('NamedEntity', 1)
# print(numeric_test_frame)
# print(numeric_train_frame)

alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
print("Traininig")

alg.fit(numeric_train_frame[list_of_predictors], numeric_train_frame["NamedEntity"])

print("Predicting with O")
predictions = alg.predict(numeric_test_frame[list_of_predictors])

print "Accuracy: ", accuracy_score(numeric_test_frame["NamedEntity"], predictions)
print "Precision: ", precision_score(numeric_test_frame["NamedEntity"], predictions, average='micro')
print "Recall: ", recall_score(numeric_test_frame["NamedEntity"], predictions, average='micro')
print "F1-Score: ", f1_score(numeric_test_frame["NamedEntity"], predictions, average='micro')

print("\n Predicting without O")
predictions = alg.predict(new_numeric_test_frame[list_of_predictors])

print "Accuracy: ", accuracy_score(new_numeric_test_frame["NamedEntity"], predictions)
print "Precision: ", precision_score(new_numeric_test_frame["NamedEntity"], predictions, average='micro')
print "Recall: ", recall_score(new_numeric_test_frame["NamedEntity"], predictions, average='micro')
print "F1-Score: ", f1_score(new_numeric_test_frame["NamedEntity"], predictions, average='micro')
