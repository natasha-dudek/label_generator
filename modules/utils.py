from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
from nltk import word_tokenize 
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
from sklearn.utils import shuffle
import torch

def tokenize(df):
	"""
	Perform tokenization of a corpus
	
	Arguments:
		df (pd.DataFrame) -- contains columns 'descriptions' with corpus entries
	
	Returns:
		tokenized_corpus (list of lists) -- tokens in each document in the corpus
	"""
	corpus_descriptions = [i.lower() for i in df['description']]
	
	tokenized_corpus = []
	
	for description in corpus_descriptions:
		token = word_tokenize(description)
		tokenized_corpus.append(token)
	
	return tokenized_corpus
	
def word_to_int(tokenized_corpus, min_count):
	"""
	Create dictionaries that map tokens in the corpus to ints and vice versa
	
	Note: we will keep only tokens 
	
	Arguments:
		 tokens (list of lists) -- tokens in each document in the corpus
		 min_count (int) -- only keep tokens that occur >= min_count times in corpus
	
	Returns:
		token_to_num (dict) -- maps each token in corpus to unique int
		num_to_token (dict) -- maps ints to unique token in corpus
	"""
	flat_tokens = [i for sublist in tokenized_corpus for i in sublist]
	token_counts = Counter(flat_tokens)
	keepers = [i for i in token_counts if token_counts[i] >= min_count]
	
	token_to_num = dict((c, i) for i, c in enumerate(keepers))
	num_to_token = {v: k for k, v in token_to_num.items()}
	
	return token_to_num, num_to_token

def n_grams(tokenized_doc, n_len, token_to_num):
	"""
	Converts a wine description into n-grams, where n = len(tokens) - n_len + 1
	
	Arguments:
		tokenized_doc (list) -- tokens in a document (i.e. a tokenized wine description)
		n_len (int) -- length of input / output sequences for the LSTM
		token_to_num (dict) -- maps each token in corpus to unique int
	
	Returns:
		x (list of lists) - n-grams for a given document, where tokens are encoded as ints
		y (list of lists) -- n-grams shift one to the right compared to x
	"""
	
	x = []
	y = []
	
	for i in range(n_len, len(tokenized_doc) - 1): # Added -1 to keep seq_size constant
		n_gram = tokenized_doc[i - n_len : i + 1] # Added +1 to keep seq_size constant
		
		# We discarded rare tokens earlier on
		# If a rare / discarded token appears in an n-gram, don't use it
		good_ngram = True
		for s in n_gram:
			if s not in token_to_num:
				good_ngram = False
	
		if good_ngram:
			x.append([token_to_num[s] for s in n_gram[:-1]])
			y.append([token_to_num[s] for s in n_gram[1:]]) 
					   
	return x, y

def load_glove(path, token_to_num, embedding_size):
	"""
	Load GloVe embeddings
	
	Arguments:
		path (str) -- path to glove embeddings (including filename)
		token_to_num (dict) -- maps each token in corpus to unique int
		embedding_size (int) -- GloVe vector dimension
		
	Returns:
			
	"""
	# Initialize embeddings 
	# For glove.6B.300d.txt, embedding values are:
	# 1. normally distributed
	# 2. Range from 3.2582 to -3.0639
	# 3. Distrib has avg = -0.00390501189253 with std = 0.3817702235076489
	embeddings = np.random.normal(-0.00390501189253, 0.3817702235076489, size=(len(token_to_num), embedding_size))
	
	with open(path) as f:		
		for line in f.readlines():
			
			token = line.split()[0]
			e_vec = line.split()[1:]
			
			if token in token_to_num:
				vector = np.array(e_vec, dtype='float32')
				embeddings[token_to_num[token]] = vector
				
		return torch.from_numpy(embeddings).float()

def make_training_set(tokenized_corpus, seq_size, token_to_num):
	"""
	Create training and test sets to use for our LSTM
	
	Arguments:
		tokenized_corpus (list of lists) -- tokens in each document in the corpus
		seq_size (int) -- length of sequences for LSTM
		token_to_num (dict) -- maps each token in corpus to unique int
		
	Returns:
		X (np.ndarray) -- training set of n-grams
		Y (np.ndarray) -- test set of n-grams
		seeds (list of lists) -- each list contains the first three words of a real wine description
	"""	
	X = []
	Y = []  
	seeds = [] 
	for tokenized_doc in tokenized_corpus:
		x, y = (n_grams(tokenized_doc, seq_size, token_to_num))
		X.extend(x)
		Y.extend(y)
	
		# If a seed has a word that isn't in our vocabulary, don't keep it
		good_seed = True
		for i in tokenized_doc[0:3]:
			if i not in token_to_num:
				good_seed = False
		if good_seed:
			seeds.append(tokenized_doc[0:3]) 
	
	X = np.array(X)
	Y = np.array(Y)
	
	# Shuffle X and Y so that when we get batches for training,
	# those batches are random selections 
	X, Y = shuffle(X, Y, random_state=0)
	
	return X, Y, seeds

def datenow():
	"""
	Get the current date and time (year, month, day, hour, minute, second)
	
	Arguments: 
		None
		
	Returns:
		(str) -- year-month-day-hour-minute-second
	"""
	now = datetime.now()
	date_list = [now.year, now.month, now.day, now.hour, now.minute, now.second]
	return '-'.join([str(i) for i in date_list])
	
def reinstate_capitalization(tokenized_doc):
	"""
	For text in all lowercase letters, will re-instate proper capitalization (first letter of first word in any sentence will be a capital)
	
	Arguments:
		tokenized_doc (list) -- tokens that make up a text (all lowercase)
	
	Returns:
		tokenized_doc (list) -- tokens that make up a text, with proper capitalization
	"""
	for i in range(0,len(tokenized_doc)):
		if i == 0: # first word in description should be capitalized
			tokenized_doc[i] = tokenized_doc[i].title()
		elif tokenized_doc[i] == "." and (i != len(tokenized_doc)-1): # words following a period should be capitalized
			tokenized_doc[i+1] = tokenized_doc[i+1].title() 
			
	return tokenized_doc

def correct_ending(tokenized_doc):
	"""
	Remove incomplete sentences at the end of a generated document (i.e. wine description)
	
	Arguments:
		tokenized_doc (list) -- tokens that make up a text
		
	Returns:
		tokenized_doc (list) -- tokens that make up a text minus any incomplete sentence at the end (or a "bad output" designation if there was no complete sentence)
	"""	
	tokenized_doc = list(map(lambda x: x.replace(" .", "."), tokenized_doc))
	
	
	if "." in tokenized_doc:
		while tokenized_doc[-1] != ".":
			 del tokenized_doc[-1]
	else:
		tokenized_doc = ['bad output']
		
	return tokenized_doc			

def good_candidates(prob_tensor, words, num_to_token, top_k):
	"""
	Selects the next word in the prediction
	
	The process:
	0. Compare the probability of each of the top_k most likely next words 
	1. Discard words in the top_k that have a probability of less than half that of the most likely next word 
	2. Randomly pick one of the remaining words
	3. If that word has already been used in the sentence and is not a "common word" (e.g. "the"), pick another word. 
		This is important for avoiding phrases like:
			It's a blend of 50% merlot , 20% cabernet sauvignon and 25% merlot, this opens with aromas of blackberry, 
				plum, prune, leather, leather and a whiff of leather.
			The palate offers dried black cherry, licorice, espresso and espresso alongside round tannins
			The palate is full and full, with a smooth, supple feel
			The palate is round, creamy and creamy
			The palate is a bit more reserved, but the palate is fresh and fresh, with a lovely ginger-infused flavor
			Pleasing aromas of vanilla and vanilla.
			this wine is rich, rich and rich
			The juicy palate delivers juicy black cherry
		
	Arguments: 
		prob_tensor (torch.tensor) -- output[0] from the neural network, contains prob of each word in vocab as next word
			Note: words are represented by their index, not by the actual word
		words (list) -- the tokens that have been chosen thus far in your output description e.g. ['a', 'fine', 'choice']
		num_to_token (dict) -- maps ints to unique token in corpus
		top_k (int) -- top k results to sample word from
	
	Returns:
 	   words (list) -- the final choice of the next word
	"""
	
	probs, top_ix = torch.topk(prob_tensor, k=top_k)
	softmax_probs = softmax(probs.tolist()[0]).tolist()
	max_prob = max(softmax_probs)
	top_ix = top_ix.tolist()[0]
	
	keepers = [] # keepers is a list of indexes of words to keep
	for i in range(len(softmax_probs)):
		if softmax_probs[i] >= max_prob/2:
			keepers.append(top_ix[i])	
	
	choice = np.random.choice(keepers) # choice is an index representing a word
	
	# If a word has already been used in a sentence, don't reuse it, unless it is a common word
	# To make common_words, I printed the top 100 most common words in the counter dictionary
	# Chose some that I think are ok if they reoccur in a single description - may take some tuning
	common_words = [',','.','and','the','a','of','with','this','is','it','in','to','on', 'by',
				   "'s",'that','from','are','has','for','but','%','as','in', 'its', 'at','or',"it's"]
	
	word_choice = num_to_token[choice] # convert index to actual word
	
	if word_choice in words and word_choice not in common_words:
		keepers.remove(choice)
		if len(keepers) > 1:
			choice = np.random.choice(keepers)
			words.append(num_to_token[choice])
		else:
			# might end up here with a phrase like "this full-bodied" -- only logical next choice is "wine"
			# take a few steps back and try again
			del words[-3:]	
	else:
		words.append(word_choice)
		
	return words

def predict(net, token_to_num, num_to_token, e, seeds, top_k):
	"""
	Use the NN to generate a wine tasting description
	
	Arguments:
		net () --
		token_to_num (dict) -- maps each token in corpus to unique int
		num_to_token (dict) -- maps ints to unique token in corpus
		e (int) -- current epoch -> val does not matter if in eval mode
		seeds (list of lists) -- each list contains the first three words of a real wine description
		top_k (int) -- top k results to sample word from
	
	Returns:
		words () --
	"""
	net.eval()

	state_h, state_c = net.zero_state(1)
	
	while True:
	
		# Randomly select a seed (list of three tokens) to start with
		random_int = np.random.randint(0, len(seeds)-1)
		words = seeds[random_int]
		
		for w in words:
			ix = torch.tensor([[token_to_num[w]]])
			output, (state_h, state_c) = net(ix, (state_h, state_c), e)
		
		# Select next word in sentence
		words = good_candidates(output[0], words, num_to_token, top_k)
	
		len_sent = np.random.randint(45,55) # this is how many words long the generated descrpiption will be
		
		try:
			choice = token_to_num[words[-1]]
		except IndexError:
			choice = token_to_num[words[-2]]
			
		for _ in range(len_sent):
			ix = torch.tensor([[choice]]) 
			output, (state_h, state_c) = net(ix, (state_h, state_c), e)
	
			words = good_candidates(output[0], words, num_to_token, top_k)
			choice = token_to_num[words[-1]]
		
		words = correct_ending(words) # Remove last "sentences" that end prematurely (i.e. without a period)		
		if len(words) < 25:
			pass
		else:
			# format words nicely
			words = reinstate_capitalization(words) # After a period there should be a capital letter
			text = " ".join(words)
			text = text.replace(" .", ".")
			text = text.replace(" ,", ",")
			text = text.replace(" %", "%")
			text = text.replace(" '", "'")
			text = text.replace(" :", ":")
			
			return text
			
def softmax(x):
	"""
	Return softmax
	
	Arguments:
		x (list) -- logits
	
	Returns:
		(list) -- softmax
	"""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()
	
def learning_curve(train_losses):
	"""
	Plot learning curve -- experience vs loss
	
	Arguments:
		train_losses (list) -- training losses 
		
	Returns:
		matplotlib.Figure
	"""
	fig = plt.figure()
	x_losses = [*range(len(train_losses))]
	plt.plot(x_losses, train_losses)
	plt.xlabel('Experience (batch)')
	plt.ylabel('Loss (CE)')
	plt.tight_layout()
		
	return fig
	
	