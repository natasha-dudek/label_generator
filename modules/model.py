import torch
import torch.nn as nn

from modules import utils

class RNNModule(nn.Module):
	"""
		An RNNModule object contains an LSTM network. During training, the LSTM intitially uses frozen pre-trained GloVe embeddings, and then later progresses to fine-tuning them on the dataset. Training is performed with dropout. 
		"""
	def __init__(self, lstm_size, path_to_glove, token_to_num, embedding_size, num_layers, drop_prob):
		super(RNNModule, self).__init__()
		self.lstm_size = lstm_size
		self.len_vocab = len(token_to_num)
		self.num_layers = num_layers
		self.drop_prob = drop_prob
		
		weight = utils.load_glove(path_to_glove, token_to_num, embedding_size)
		self.embedding1 = nn.Embedding.from_pretrained(weight, freeze=True)
		self.embedding2 = nn.Embedding.from_pretrained(weight)
		
		self.lstm = nn.LSTM(embedding_size, self.lstm_size, \
							num_layers=self.num_layers, dropout=self.drop_prob)
		self.dense = nn.Linear(self.lstm_size, self.len_vocab)
	
	def forward(self, x, prev_state, e):
		"""
		Take an input sequence + the previous state, produce output + current state.
		
		Note: In the early stages of training, use frozen GloVe embeddings. Later on do fine-tuning of embeddings.
		
		Arguments:
			x (torch.tensor) -- batch of input sequences
			prev_state (torch.tensor, torch.tensor) -- previous state hidden & current
			e (int) -- current epoch
		
		Returns:
			logits (torch.tensor) -- unnormalized predictions
			state (torch.tensor) -- current state
		"""
		if e < 3: # use frozen embeddings for first 3 epochs
			embed = self.embedding1(x) 
		else: # progress to fine-tuning embeddings
			embed = self.embedding2(x)
			
		output, state = self.lstm(embed, prev_state)
		logits = self.dense(output)
		
		return logits, state

	def zero_state(self, batch_size):
		"""
		Resets states of LSTM to zero
		
		Arguments:
			batch_size (int) -- number of instances in a batch
		
		Returns:
			(torch.tensor, torch.tensor) -- zeroed states   	
		"""
		return (torch.zeros(self.num_layers, batch_size, self.lstm_size),
				torch.zeros(self.num_layers, batch_size, self.lstm_size))