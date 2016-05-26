import numpy as np
def fromCorpusToRaw(filenameInput, filenameOutput):
	entity_tags = {}
	targets = {}
	outFile = open(filenameOutput, 'w')
	i = 0
	with open(filenameInput) as inFile:
		for line in inFile:
			i = i + 1
			if "DOCSTART" not in line:
				if len(line.strip()) != 0:
					words = line.split()
					word = words[0]
					if word not in " '.,:;()=\"/\\" and '-' not in word:
						if not i==1:
							outFile.write(" ")
						entity_tags[words[3]] = ""
						targets[word] = words[3]
						outFile.write(word)
	return entity_tags.keys(), targets

def loadDataset(model, entity_tags, targets_entity):
	target_values = {}
	target_values = {'I-LOC': 0, 'B-ORG': 1, 'O': 4, 'I-PER': 3, 'I-MISC': 2, 'B-MISC': 2, 'I-ORG': 1, 'B-LOC': 0}

	dataset = model.vectors[1:]
	targets = np.empty(dataset.shape[0],).astype(int)
	for i in range(1, len(model.vocab)):
	    targets[i-1] = target_values[targets_entity[model.vocab[i]]]
	return dataset, targets, target_values