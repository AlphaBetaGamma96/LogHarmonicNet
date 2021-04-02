#Python packages
import torch
import torch.nn as nn
from time import time
import sys, gc

#Custom classes
from Models import LogHarmonicNet
from Samplers import RandomWalkMetropolisHastings as RWMH
from Hamiltonian import HarmonicOscillatorWithInteraction1D as HOwInt1D
from VMC import ADVMC
from utils import memory_usage, calc_clip
from Writer import WriteToFile

#Hardware Hyperparameters
device = torch.device('cpu')

#Gaussian Interaction Hyperparameters
V0 = -3
sigma0 = 0.5

#Network Hyperparameters
nfermions = 2   #number of input nodes
num_hidden = 16  #number of hidden nodes per layer
num_layers = 1   #number of layers in network
num_dets = 1     #number of determinants (currently only accepts 1)
func = nn.Tanh() #activation function between layers

#Feed-Forward Neural Network which is a R^N -> R^1 function. It returns 
#the log. abs. determinant of network's output (along with its sign)
net = LogHarmonicNet(num_input=nfermions,
                     num_hidden=num_hidden,
                     num_layers=num_layers,
                     num_dets=num_dets,
                     func=func)
net=net.to(device)
net=torch.jit.script(net)

#Sampler Hyperparameters  (Markov-Chain Monte Carlo)
nblocks = 50             #number of blocks in Markov Chain
nsteps = 50              #number of samples within a given block
nsamples = nblocks*nsteps #total number of samples within a Markov Chain
burn_in = 0               #number of burn in samples of Markov Chain
thinning = 10             #thinning factor for samples (take every thinning-th sample, i.e. if thinning = 10, take every 10-th sample from chain)
nwalkers = 100            #number of chains/walkers 
std = 1                 #width of the proposal distribution

sampler = RWMH(network=net,
               dim=nfermions,
               nwalkers=nwalkers,
               std=std,
               device=device)

#Analtyical Solution (lowest loss value for given value of 'nfermions')
groundstate = nfermions**2*(40.0*nfermions**(-1/3))/2.0
               
calc_local_energy = HOwInt1D(network=net, V0=V0, sigma0=sigma0) #class that calculates the local matrix elements of the Hamiltonian
                         
advmc_estimator = ADVMC(network=net) #class that combines all local matrix elements together to return mean loss (and its error)

optim = torch.optim.Adam(net.parameters(), #optimiser class
                         lr=1e-3,
                         betas=(0.9,0.999),
                         eps=1e-8)


#file name to save data of interest
fname = "A%02i_H%03i_B%03i_S%03i_V0_%04.2f_Sig0_%04.2f.csv" % (nfermions, num_hidden, nblocks, nsteps, V0, sigma0)
writer = WriteToFile(load=None, filename=fname)


net.train()

epochs=10000 #number of epochs

prefix='GB' #for memory usage

for epoch in range(epochs+1):
  
  stats={} #dict to store items of interest
  
  start=time() #record time per epoch
  
  mem1=memory_usage(prefix) #record current ram usage 
  
  X, acceptance = sampler(nsamples=nsamples, #sampler object which returns samples (X) and their acceptance rate 
                          burn_in=burn_in,   #from the Markov Chain. X is of shape [nwalkers, nsamples, dim]
                          thinning=thinning)
                          
  mem2=memory_usage(prefix) #record current ram usage 
  
  E_local = calc_local_energy(X) #Calculate local energy from the samplers (X) returns shape [nwalkers, nsamples]
  
  mem3=memory_usage(prefix) #record current ram usage 
  
  energy_mean, energy_std, sampler_stats = advmc_estimator(X, E_local, nblocks) #Takes initial samples, and local energy and computes 
                                                                                #the total energy (which the loss of our choice)
  mem4=memory_usage(prefix) #record current ram usage 
  
  optim.zero_grad(set_to_none=True) #zero gradient cache 
  energy_mean.backward()            #calculate loss gradients 
  optim.step()                      #update parameters
  
  mem5=memory_usage(prefix) #record current ram usage 

  end = time() #record time per epoch

  #record items of interest into a dict, which is written to pandas dataframe.
  stats['epoch'] = [epoch]
  stats['energy_mean'] = energy_mean.item()
  stats['energy_std'] = energy_std.item()
  stats['groundstate'] = groundstate
  stats['envelope_width'] = net.width.item()
  stats = {**stats, **sampler_stats} #merge dicts
  stats['ram_sample'] = mem2-mem1                  #memory usage before/after the calling of certain classes
  stats['ram_local'] = mem3-mem2
  stats['ram_advmc'] = mem4-mem3
  stats['ram_optim'] = mem5-mem4
  stats['ram_epoch'] = mem5-mem1
  stats['ram_total'] = mem5
  stats['walltime'] = end-start
  
  writer(stats) #write to file...

  #print some useful information during runtime
  print("Epoch: %6i | Energy: %4.2f +/- %4.2f MeV | GS: %4.2f MeV | Walltime: %4.2e%s | RAM: %4.2e%s (%+4.2e%s) | Sample: %4.2e%s Local: %4.2e%s ADVMC: %4.2e%s Optimiser: %4.2e%s" % (epoch, energy_mean, energy_std, groundstate,end-start,"s", memory_usage(prefix), prefix, mem5-mem1,prefix, mem2-mem1,prefix,mem3-mem2,prefix,mem4-mem3,prefix,mem5-mem4,prefix))

















