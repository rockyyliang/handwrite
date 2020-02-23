
import sys
sys.path.insert(0, '..')
import numpy as np

import torch

from modules import WRITER
from sampler import sample_mix_bernoulli, sample_mix_gaussian
    
'''define device'''
if torch.cuda.is_available():
	DEVICE = torch.device('cuda:0')
else:
	DEVICE = torch.device('cpu')

def generate_unconditionally(timesteps=500):
	'''unconditional run'''
	writer = WRITER().double().to(DEVICE)
	
	writer.load_state_dict(torch.load('../models/final_weights/final_uncond.pt'))
	writer.eval()

	#initialize first input
	X = torch.zeros(1,1,3)
	#X[:,:,1:] = -0.01*torch.ones(1,2)
	X[:,:,1:] = -0.01*torch.randn(1,2)
	X = X.double().to(DEVICE)

	with torch.no_grad():
		pos,end,hidden_states = writer(X)

	eos = sample_mix_bernoulli(end[0], end[1])
	step = sample_mix_gaussian(pos[0], pos[1], pos[2])

	X = torch.zeros(1,1,3)
	X[:,:,0] = eos
	X[:,:,1:] = step
	X = X.double().to(DEVICE)

	be = []
	dx = []
	dy = []
	for ts in range(timesteps):
		with torch.no_grad():
		    pos, end, hidden_states = writer(X, hidden_states)
		
		eos = sample_mix_bernoulli(end[0], end[1])
		step = sample_mix_gaussian(pos[0], pos[1], pos[2])
		X = torch.zeros(1,1,3)
		X[:,:,0] = eos
		X[:,:,1:] = step
		X = X.double().to(DEVICE)
		
		be.append(eos)
		dx.append(step[0])
		dy.append(step[1])

	stroke = np.column_stack((be, dx, dy))
	return stroke
