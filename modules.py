import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MDN_G(nn.Module):
    def __init__(self, in_dim, out_dim, n_dist):
        '''
        gaussian mdn
        n_dist: how many distributions for each out_dim
        '''
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_dist = n_dist

        self.pi = nn.Linear(self.in_dim, self.n_dist)
        self.sigma = nn.Linear(self.in_dim, self.out_dim*self.n_dist)
        self.mu = nn.Linear(self.in_dim, self.out_dim*self.n_dist)

    def forward(self, x):
        pi = F.softmax(self.pi(x),dim=1)

        #reshape to (batch, dist, out)
        sigma = torch.exp(self.sigma(x)).view(-1, self.n_dist, self.out_dim)
        mu = self.mu(x).view(-1, self.n_dist, self.out_dim)

        return pi, sigma, mu

class MDN_B(nn.Module):
    def __init__(self, in_dim, out_dim, n_dist):
        '''
        bernoulli mdn
        n_dist: how many distributions for each out_dim
        '''
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_dist = n_dist

        self.pi = nn.Linear(self.in_dim, self.n_dist)
        self.probs = nn.Linear(self.in_dim, self.out_dim*self.n_dist)

    def forward(self, x):
        epsilon = 1e-6
        pi = F.softmax(self.pi(x),dim=1)

        #reshape to (batch, dist, out)
        #sigmoid is applied in loss function for stability
        probs = self.probs(x) + epsilon
        probs = probs.view(-1, self.n_dist, self.out_dim)
        return pi, probs

class WRITER(nn.Module):
    def __init__(self, hidden_dim=450, n_dist=20):
        '''
        unconditional prediction network
        '''
        super().__init__()
        self.n_dist = n_dist
        #alex graves' paper used a hidden size of 900
        #but his dataset is roughly twice the size
        #so we'll use 450
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size = 3,
            hidden_size = self.hidden_dim,
            num_layers = 1,
            batch_first = True,
            dropout = 0,
            bidirectional = False
        )

        self.adapter = nn.Linear(self.hidden_dim, self.hidden_dim)

        #position mdn outputs x and y position normal distribution mix
        #end of stroke mdn outputs eos bernoulli logit probability mix
        self.mdn_position = MDN_G(self.hidden_dim, 2, n_dist=self.n_dist)
        self.mdn_end = MDN_B(self.hidden_dim, 1, n_dist=self.n_dist)

    def forward(self, x, hidden_states=None):
        x, hidden_states = self.lstm(x, hidden_states)
        #print(x[:,-1,:].shape)
        #print(hidden_states[0])
        x = F.relu(self.adapter(x[:,-1,:]))
        pos = self.mdn_position(x)
        end = self.mdn_end(x)
        return pos, end, hidden_states

class WRITER_COND(nn.Module):
    def __init__(self, hidden_dim=450, vocab_dim=100, n_dist=20, device=torch.device('cpu')):
        super().__init__()
        self.n_dist = n_dist
        self.hidden_dim = hidden_dim
        self.attention_out_dim = n_dist*3
        self.vocab_dim = vocab_dim

        #setup constants
        self.device = device
        self._init_constants()

        #input size for final lstm
        self.lstm_1_in_dim = 3 + hidden_dim + vocab_dim

        #first lstm layer that processes stroke input
        self.lstm_0 = nn.LSTM(
            input_size = 3,
            hidden_size = self.hidden_dim,
            num_layers = 1,
            batch_first = True,
            dropout = 0,
            bidirectional = False
        )

        #attention layer
        #output is n_dist times 3 because we need
        #a set of (alpha, beta, k) for every distribution
        self.attention = nn.Linear(self.hidden_dim, self.attention_out_dim)

        #2nd lstm
        #this layer takes in input(skip connection), last lstm's hidden, and encoded text
        self.lstm_1 = nn.LSTM(
            input_size = self.lstm_1_in_dim,
            hidden_size = self.hidden_dim,
            num_layers = 1,
            batch_first = True,
            dropout = 0,
            bidirectional = False
        )

        #mdn heads
        self.mdn_position = MDN_G(self.hidden_dim, 2, n_dist=self.n_dist)
        self.mdn_end = MDN_B(self.hidden_dim, 1, n_dist=self.n_dist)

    def _init_constants(self, device=None):
        '''set constant device'''
        #allow device to be passed in as arg
        #if not given, use default
        if device==None:
            device = self.device

        self.u = torch.arange(0,self.vocab_dim).view(1,1,-1).to(device)



    def forward(self, x, text, k_integral=0, hidden_states=(None, None)):
        '''
        x: last timestep
        text: sentence encoded at the character level
        k_integral: accumulated k
        hidden_states: tuple of hidden states for lstm_0 and lstm_1
        '''
        #get sequence length
        sequence_len = x.shape[1]

        #unpack hidden states
        hidden_0 = hidden_states[0]
        hidden_1 = hidden_states[1]

        self.lstm_0.flatten_parameters()
        lstm_out_0, hidden_0 = self.lstm_0(x, hidden_0)
        attention_out = torch.exp(self.attention(lstm_out_0)).view(-1, sequence_len, self.n_dist, 3, 1)

        a = attention_out[:,:,:,0]
        b = attention_out[:,:,:,1]
        k = attention_out[:,:,:,2]


        #accumulate k
        if k_integral==0:
            k_use = k.cumsum(dim=1)
        else:
            k_use = k_integral.unsqueeze(1) + k

        #u = torch.arange(0,self.vocab_dim).view(1,1,-1)

        #weights for encoded chars
        phi = (torch.exp(-b * (k_use - self.u).pow(2)) * a).sum(-2)

        #multiply chars with weights
        w = torch.matmul(phi, text)
        #print(w.shape)

        self.lstm_1.flatten_parameters()
        lstm_in_1 = torch.cat([x, lstm_out_0, w],dim=-1)
        #print(lstm_in_1.shape)
        lstm_out_1, hidden_1 = self.lstm_1(lstm_in_1, hidden_1)

        #get last timestep from lstm_1 and feed to mdn
        final_timestep = lstm_out_1[:,-1,:]
        pos = self.mdn_position(final_timestep)
        end = self.mdn_end(final_timestep)

        return pos, end, (hidden_0, hidden_1)

'''
below are loss functions

TODO:
mdn_loss_gaussian is for n amount of independent variables
Graves paper uses bivariate, try that if time allows
def gaussian_probability_bivariate()
def mdn_loss_bivariate()
'''
def gaussian_probability(sigma, mu, target):
    target = target.unsqueeze(1).expand_as(sigma)
    ret = (1/math.sqrt(2*math.pi)) * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)

def mdn_loss_gaussian(pi, sigma, mu, target):
    prob = pi * gaussian_probability(sigma, mu, target)
    #print(prob.shape)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)

def mdn_loss_bernoulli(pi, probs, target, pos_weight):
    #apply weight pi to logits then calc bce loss
    probs = probs.squeeze()
    weighted_probs = probs*pi

    return F.binary_cross_entropy_with_logits(
        input=weighted_probs.mean(dim=1),
        target=target.squeeze(),
        pos_weight=pos_weight
    )
