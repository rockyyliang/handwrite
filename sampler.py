'''
data utilities
'''
import numpy as np

import torch
from torch.utils.data import Dataset

CHAR_BANK = ' abcdefghijkilmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"-.,\'!:?#()'

class SeriesSampler(Dataset):
    def __init__(self, texts, strokes, sequence_len=30, val_split=0.8, val=False, cond=False):
        '''
        Dataset class used during training

        texts: list of strings
        strokes: np array of shape (datapoints, timestep, 3)
        val: boolean, true if you wish to generate validation data
        cond: boolean, true if training conditional model
        '''
        self.texts = texts
        self.strokes = strokes
        self.num_strokes = len(strokes)
        self.cond = cond

        #find total number of datapoints
        self.stroke_lengths = []
        for s in strokes:
            #loop through strokes
            self.stroke_lengths.append(len(s))

        #integral array is used in _locate_stroke()
        self.stroke_lengths_integral = np.cumsum(self.stroke_lengths)

        #how many timesteps there are
        self.num_datapoints = self.stroke_lengths_integral[-1]

        #validation mode or not
        self.val = val
        self.val_split = val_split
        self.val_begin_idx = int(val_split*self.num_datapoints)

        #length of temporal sequence
        self.sequence_len = sequence_len



    def __len__(self):
        '''return length of usable data'''
        if self.val:
            #validation mode
            total_timesteps =  self.num_datapoints - self.val_begin_idx
        else:
            #training mode
            total_timesteps = self.val_begin_idx
        return total_timesteps

    def __getitem__(self, idx):
        '''calls __generate_data()'''
        if self.cond:
        	X, enc, y = self.__generate_data(idx)
        else:
        	X, y = self.__generate_data(idx)


        #convert to torch tensor
        #use as_tensor() indstead of tensor() because it shares memory
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)
        if self.cond:
        	enc = torch.as_tensor(enc)
        	return X, enc, y
        else:
        	return X, y

    def __generate_data(self, idx):
        '''
        takes an index, return input and output in np arrays
        FIX COMMENT BELOW

        '''
        #index of label out of ALL timesteps
        if self.val:
            label_idx = idx + self.val_begin_idx
        else:
            label_idx = idx

        stroke_idx = self._locate_stroke(label_idx)
        #print(stroke_idx)
        label_idx_adjusted = label_idx
        if stroke_idx > 0:
            #subtract integral value and get index of this timestep within this stroke
            label_idx_adjusted = label_idx - self.stroke_lengths_integral[stroke_idx-1]-1
        if label_idx_adjusted == self.stroke_lengths_integral[0]:
            #hacky fix for end of first stroke
            label_idx_adjusted -= 1
        if label_idx_adjusted == 0:
            #can't sample first timestep of each stroke
            label_idx_adjusted = 1
        #print('adjusted label:', label_idx_adjusted)


        y = self.strokes[stroke_idx][label_idx_adjusted]

        #create and size np array for inputs
        #shape: sequence length x 3
        X = np.zeros((self.sequence_len, 3))

        #iterate backwards from label to populate X with input sequence
        #there is 'label_idx_adjusted' amount of timesteps available as input
        for n, ts in enumerate(range(self.sequence_len-1, -1, -1)):
            #start from label_idx_adjusted-1, decrement until fully populated
            #or we hit the beginning of stroke. in this case, arrays are zero padded
            label_idx_input = label_idx_adjusted-n-1
            #print(label_idx_input)

            X[ts] = self.strokes[stroke_idx][label_idx_input]
            if label_idx_input <= 0:
                break

        if not self.cond:
            return X, y
        else:
            text = self.texts[stroke_idx]

            enc = make_encoded_input(text)
            #print(text)
            return X, enc, y

    def _locate_stroke(self, idx):
        '''
        given timestep index, find index of self.strokes it belongs to
        '''
        #print('searching', idx)
        i = 0
        j = self.num_strokes-1
        while(i<j):
            if self.stroke_lengths_integral[i] >= idx:
                return i
            if self.stroke_lengths_integral[j] < idx:
                return j+1

            i += 1
            j -= 1

        return i
        #in case something goes wrong above
        #print('two pointer search failed, i: {}, j: {}, idx: {}'.format(i, j, idx))
        #return -1

def str2onehot(string):
    '''
    input: text string
    output: numpy array of input's one hot encoding
    '''
    char_bank = CHAR_BANK
    #print(len(char_bank))
    bank_size = len(char_bank)
    str_length = len(string)

    #initialize output. set dtype to int8 to make it smaller
    onehot = np.zeros((bank_size, str_length), dtype=np.int8)

    for n, c in enumerate(string):
        char_idx = char_bank.find(c)
        if char_idx == -1:
            #chars not in the bank
            #treat it as space
            char_idx = 0
        onehot[char_idx, n] = 1

    return onehot

def make_encoded_input(string, length=65):
	'''
	cut or pads encoding to be the same length
	outputs np array with set length
	'''
	encoding = str2onehot(string)
	encoding_fixed_length = np.zeros((len(CHAR_BANK), length))

	if encoding.shape[1] <= length:
		#zero pad text string that's too short
		encoding_fixed_length[:,:encoding.shape[1]] = encoding
	else:
		#exceeded set length
		#trim encoding
		encoding_fixed_length = encoding[:,:length]
	return encoding_fixed_length

'''
below are functions used to sample from distribution mixtures
'''

def sample_mix_gaussian(pi, sigma, mu, bias=0, return_torch=True):
    '''
    given parameters of gaussian mixture, sample from it

    INPUTS
    pi: distribution weights (1 x n_dist)
    sigma: variance (1 x n_dist x out_dim)
    mu: mean (1 x n_dist x out_dim)

    can only handle batch of one for now
    '''
    #squeeze inputs here
    pi = pi.squeeze()
    sigma = sigma.squeeze()
    mu = mu.squeeze()

    pi_adjusted = torch.nn.functional.softmax(torch.log(pi+bias), dim=0)

    #convert to np
    if str(type(pi_adjusted)) == "<class 'torch.Tensor'>":
        if pi.is_cuda:
            pi_adjusted = pi_adjusted.cpu().detach().numpy()
    if str(type(sigma)) == "<class 'torch.Tensor'>":
        if sigma.is_cuda:
            sigma = sigma.cpu().detach().numpy()
    if str(type(mu)) == "<class 'torch.Tensor'>":
        if mu.is_cuda:
            mu = mu.cpu().detach().numpy()

    n_dist = len(pi)
    dist_idx_range = np.arange(n_dist)


    #choose which distribution to use based on pi array
    idx = np.random.choice(dist_idx_range, p=pi_adjusted)


    #sample from chosen mu and sigma
    #adjust sigma for bias
    sigma_adjusted = np.exp(np.log(sigma[idx]) - bias)
    sample = np.random.normal(mu[idx], sigma[idx])
    if return_torch:
        return torch.tensor(sample)
    return sample

def sample_mix_bernoulli(pi, probs, return_torch=True):
    '''
    given parameters of bernoulli mixture, sample from it

    INPUTS
    pi: distribution weights (1 x n_dist)
    probs: positive probability (1 x n_dist x out_dim)

    can only handle batch of one for now
    '''
    #squeeze inputs here
    pi = pi.squeeze()
    probs = probs.squeeze()

    #convert to np
    if str(type(pi)) == "<class 'torch.Tensor'>":
        if pi.is_cuda:
            pi = pi.cpu().detach().numpy()
    if str(type(probs)) == "<class 'torch.Tensor'>":
        if probs.is_cuda:
            probs = probs.cpu().detach().numpy()

    n_dist = len(pi)
    dist_idx_range = np.arange(n_dist)

    #choose which distribution to use based on pi array
    idx = np.random.choice(dist_idx_range, p=pi)

    bernoulli = torch.distributions.bernoulli.Bernoulli(logits=probs[idx])
    sample = bernoulli.sample()
    if not return_torch:
        return sample.numpy()
    return sample

if __name__ == '__main__':
	#test dataset class

	pass
