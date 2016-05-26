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

dict_lines = [line.rstrip('\n') for line in open('english_dict.txt')]
eng_dict = {}
for line in dict_lines :
    eng_dict[line] = 1

def ne_features(sentence, i, history):
     word, pos = sentence[i]
     if i == 0:
         prevword, prevword2, prevword3 = "<START>", "<START>", "<START>"
     elif i == 1:
         prevword2, prevword3 = "<START>", "<START>"
         prevword, prevpos = sentence[i - 1]
     elif i == 2:
         prevword3 = "<START>"
         prevword, prevpos = sentence[i - 1]
         prevword2, prevpos2 = sentence[i - 2]
     else:
         prevword, prevpos = sentence[i-1]
         prevword2, prevpos2 = sentence[i-2]
         prevword3, prevpos3 = sentence[i-3]

     if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
         nextword2, nextpos2 = "<END>", "<END>"
         nextword3, nextpos3 = "<END>", "<END>"
     elif i == len(sentence) - 2:
         nextword, nextpos = sentence[i+1]
         nextword2, nextpos2 = "<END>", "<END>"
         nextword3, nextpos3 = "<END>", "<END>"
     elif i == len(sentence) - 3:
         nextword, nextpos = sentence[i + 1]
         nextword2, nextpos2 = sentence [i + 2]
         nextword3, nextpos3 = "<END>", "<END>"
     else :
         nextword, nextpos = sentence[i + 1]
         nextword2, nextpos2 = sentence [i + 2]
         nextword3, nextpos3 = sentence[i+3]

     if word.istitle() :
         isUppercase = 1
     else :
         isUppercase = 0

     if word.isupper() :
         isAllUpper = 1
     else :
         isAllUpper = 0

     if any(i.isdigit() for i in word) :
         hasDigits = 1
     else:
         hasDigits = 0

     if word in eng_dict:
         inDictionary = 1
     else:
         inDictionary = 0

     return {"pos": pos,
             "word": word,
             "prevword" : prevword,
             "prevword2" : prevword2,
             "prevword3" : prevword3,
             "nextword" : nextword,
             "nextword2" : nextword2,
             "isUppercase" : isUppercase,
             "isAllUpper" : isAllUpper,
             "hasDigits" : hasDigits,
             "inDictionary" : inDictionary}

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


tp_per = 0
fp_per = 0
tn_per = 0
fn_per = 0

tp_org = 0
fp_org = 0
tn_org = 0
fn_org = 0

tp_loc = 0
fp_loc = 0
tn_loc = 0
fn_loc = 0

for s, sent in enumerate(tagged_test):
    for i, word in enumerate(sent):
        r = classifier.classify(ne_features(untag_sent(sent), i, []))
        if s <= 14:
            print(word[0][0], ' -> ', r)
        c = test_set[s][i][3]

        if r == 'I-PER':
            if r == c:
                tp_per += 1.0
            else: fp_per += 1.0
        else :
            if c != 'I-PER':
                tn_per += 1.0
            else:
                fn_per += 1.0

        if r == 'I-ORG':
            if r == c:
                tp_org += 1.0
            else:
                fp_org += 1.0
        else:
            if c != 'I-ORG':
                tn_org += 1.0
            else:
                fn_org += 1.0

        if r == 'I-LOC':
            if r == c:
                tp_loc += 1.0
            else:
                fp_loc += 1.0
        else:
            if c != 'I-LOC':
                tn_loc += 1.0
            else:
                fn_loc += 1.0


print ("FN Per %.2f" % fn_per)
precision_per =  (tp_per) / (tp_per + fp_per)
recall_per = tp_per / (tp_per + fn_per)
accuracy_per = (tp_per + tn_per) / (tp_per + tn_per + fp_per + fn_per)
f_score_per = 2 * (precision_per * recall_per / (precision_per + recall_per))

precision_org =  (tp_org) / (tp_org + fp_org)
recall_org = tp_org / (tp_org + fn_org)
accuracy_org = (tp_org + tn_org) / (tp_org + tn_org + fp_org + fn_org)
f_score_org = 2 * (precision_org * recall_org / (precision_org + recall_org))

precision_loc =  (tp_loc) / (tp_loc + fp_loc)
recall_loc = tp_loc / (tp_loc + fn_loc)
accuracy_loc = (tp_loc + tn_loc) / (tp_loc + tn_loc + fp_loc + fn_loc)
f_score_loc = 2 * (precision_loc * recall_loc / (precision_loc + recall_loc))

print("Precision Person: %.5f" % precision_per )
print("Recall Person: %.5f" % recall_per)
print("F-score Person: %.5f" % f_score_per)
print("------")
print("Precision ORG: %.5f" % precision_org )
print("Recall ORG: %.5f" % recall_org)
print("F-score ORG: %.5f" % f_score_org)
print("------")
print("Precision LOC: %.5f" % precision_loc )
print("Recall LOC: %.5f" % recall_loc)
print("F-score LOC: %.5f" % f_score_loc)

classifier.show_most_informative_features(5)