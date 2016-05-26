import nltk

def get_sentences(file):
    lines = [line.rstrip('\n') for line in open(file)]

    sentences = []
    sentence = []
    for line in lines:
        if line != '':
            sentence.append(line.split())
        else:
            sentences.append(sentence)
            sentence = []
    return sentences

def untag_sent(sentence):
    untagged_sent = []
    for ((word, tag), c, n) in sentence:
        untagged_sent.append((word, tag))
    return untagged_sent

sentences = get_sentences('eng.train.txt')



def ne_features(sentence, i, history):
     word, pos = sentence[i]
     if i == 0:
         prevword, prevpos = "<START>", "<START>"
     else:
         prevword, prevpos = sentence[i-1]
     if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
     else:
         nextword, nextpos = sentence[i+1]
     return {"pos": pos,
             "word": word,
             "prevpos": prevpos,
             "nextpos": nextpos,
             "prevpos+pos": "%s+%s" % (prevpos, pos),
             "pos+nextpos": "%s+%s" % (pos, nextpos),
             "tags-since-dt": tags_since_dt(sentence, i)}

def tags_since_dt(sentence, i):
     tags = set()
     for word, pos in sentence[:i]:
         if pos == 'DT':
             tags = set()
         else:
             tags.add(pos)
     return '+'.join(sorted(tags))

test_set = sentences[10000:]
train_set = sentences[:10000]

print ("Test set length : %d" % len(test_set))
print ("Train set length : %d" % len(train_set))

tagged_sents = [[((w, t), c, n) for [w, t, c, n] in
                 (sent)]
                for sent in train_set]


print (tagged_sents[0:5])
tagged_test = [[((w, t), c, n) for [w, t, c, n] in
                 (sent)]
                for sent in test_set]

train_set1 = []
for tagged_sent in tagged_sents:
    untagged_sent = untag_sent(tagged_sent)
    history = []
    for i, (word, c, tag) in enumerate(tagged_sent):
        featureset = ne_features(untagged_sent, i, history)
        train_set1.append((featureset, tag))
        history.append(tag)

test_set1 = []
for tagged_sent in tagged_test:
    untagged_sent = untag_sent(tagged_sent)
    history = []
    for i, (word, c, tag) in enumerate(tagged_sent):
        featureset = ne_features(untagged_sent, i, history)
        test_set1.append((featureset, tag))
        history.append(tag)

#print(train_set)

classifier = nltk.NaiveBayesClassifier.train(train_set1)

print (tagged_test[2])

correct = 0
total = 0
for s, sent in enumerate(tagged_test):
    for i, word in enumerate(sent):
        r = classifier.classify(ne_features(untag_sent(sent), i, []))
        c = test_set[s][i][3]
        if c != 'O':
            total += 1.0
            if r == c:
                correct += 1.0

print ("Correct : " , correct)
print ("Total : ", total)
print("Accuracy : %.5f" % (float(correct) / float(total)))


#print(nltk.classify.accuracy(classifier, test_set1))

classifier.show_most_informative_features(5)