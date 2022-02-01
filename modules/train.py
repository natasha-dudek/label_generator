import random

from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import torch
import torch.nn as nn

from modules import config, model, utils

def get_batches(X, Y, batch_size):
	"""
	Make batches for training
	
	Arguments:
		X (np.ndarray) -- training set of n-grams
		Y (np.ndarray) -- test set of n-grams
		batch_size (int) -- # instances per batch
	
	Yields:
		(np.ndarray, np.ndarray) -- batch of X (input) and matching batch of Y (target)
	"""
	num_batches = X.shape[0] // batch_size
	for i in range(0, num_batches*batch_size, batch_size):
		yield X[i:i+batch_size, :].T, Y[i:i+batch_size, :].T	 

def criterion_optimizer(net, learning_rate):
	"""
	Get neural net loss criterion and optimizer
	
	Arguments:
		net (RNNModule) -- LSTM of class RNNModule
		learning_rate (float) -- learning rate
	
	Returns:
		criterion () -- Cross entropy loss
		optimizer () -- Adam optimizer
	"""
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

	return criterion, optimizer

def save_model(net, train_losses, name):
	"""
	Save trained model and associated data
	
	Arguments:
		net (RNNModule) -- trained model
		train_losses (list) -- training losses 
		name (str) -- path + unique prefix by which name saved files
	"""
	print("Saving model as", name+"_model.pt")
	torch.save(net.state_dict(), name+"_model.pt")
	print("Saving training losses as", name+"_train_losses.pt")
	torch.save(train_losses, name+"_train_losses.pt")
	print("Done saving")

def load_model(model_name, losses_name, token_to_num):
	"""
	Loads model and associated data from file
	
	Arguments:
		name (str) -- path + common prefix of saved files to load 
		lstm_size (int) -- hidden size of LSTM
		path_to_glove (str) -- path to GloVe embeddings
		token_to_num (dict) -- maps each token in corpus to unique int
		embedding_size (int) -- embedding hidden size
		num_layers (int) -- number of layers in LSTM
		drop_prob (float) -- dropout probability
		
	Returns:
		net (RNNModule) -- trained model
		train_losses (list) -- training losses (KLD + BCE)
	"""
	
	hps = model_name.split("_")
	lstm_size = int(hps[8].split("size")[1])
	path_to_glove = config.GLOVE
	embedding_size = int(hps[9].split("embed")[1])
	num_layers = 3 # only coded up option to have 3 layers -- could definitely be improved
	drop_prob = float(hps[7].split("drop")[1])
	
	net = model.RNNModule(lstm_size, path_to_glove, token_to_num, embedding_size, num_layers, drop_prob)
	net.load_state_dict(torch.load(model_name))
	train_losses = torch.load(losses_name)
	
	return net, train_losses

def train(learning_rate, num_epochs, X, Y, batch_size, net, gradients_norm, token_to_num, seeds, num_to_token, top_k):
	"""
	Train LSTM of class RNNModule
	
	Arguments:
		learning_rate (float) -- learning rate
		num_epochs (int) -- total number of epochs
		X (np.ndarray) -- training set of n-grams
		Y (np.ndarray) -- test set of n-grams
		batch_size (int) -- number of instances in a batch
		net (RNNModule) -- untrained LSTM module of class RNNModule
		gradients_norm (int) -- norm to clip gradients
		token_to_num (dict) -- maps each token in corpus to unique int
		seeds (list of lists) -- each list contains the first three words of a real wine description
		num_to_token (dict) -- maps ints to unique token in corpus
		top_k (int) -- top k results to sample word from
		
	Returns:
		net (RNNModule) -- trained LSTM module of class RNNModule
		train_losses (list) -- loss after every 100 iterations
	"""
	iteration = 0
	train_losses = []
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Define loss criterion
	criterion = nn.CrossEntropyLoss()
	
	# Define optimizer
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
	
	for e in range(num_epochs):
		batches = get_batches(X, Y, batch_size)
		state_h, state_c = net.zero_state(batch_size) # reset state
		state_h = state_h.to(device)
		state_c = state_c.to(device)
	
		for x, y in batches:
			iteration += 1
	
			# Specify that we are in training mode
			net.train()
	
			# Reset all gradients
			optimizer.zero_grad()
	
			# Transfer data to device
			x = torch.tensor(x).to(device)
			y = torch.tensor(y).to(device)
	
			# Perform forward propagation
			logits, (state_h, state_c) = net(x, (state_h, state_c), e)
			
			# Calculate loss
			loss = criterion(logits.transpose(1, 2), y)
			
			# Detach states
			state_h = state_h.detach()
			state_c = state_c.detach()
	
			# Perform back-propagation
			loss.backward()
			
			# Get value for loss
			loss_value = loss.item()
			
			# Do gradient clipping
			_ = torch.nn.utils.clip_grad_norm_(net.parameters(), gradients_norm)
			
			# Update the network's parameters
			optimizer.step()
	
			# Every once in a while, print an update
			if iteration % 100 == 0:
				train_losses.append(loss_value)
				print('Epoch: {}/{}'.format((e+1), num_epochs),
					  'Iteration: {}'.format(iteration),
					  'Loss: {}'.format(loss_value))
	
			if iteration % 1000 == 0:
				seed_word = random.choice(list(token_to_num))
				test = utils.predict(net, token_to_num, num_to_token, e, seeds, top_k)
				words = TreebankWordDetokenizer().detokenize(test)
				print(test)
				net.train()
				
	return net, train_losses