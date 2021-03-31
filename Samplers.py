import torch
import torch.nn as nn

class RandomWalkMetropolisHastings(nn.Module):

  def __init__(self, network, dim, nwalkers, std, device='cpu'):
    super(RandomWalkMetropolisHastings, self).__init__()
    
    self.network = network
    if(dim is None):
      self.dim = self.network.input_layer.in_features
    else:
      self.dim = dim
    
    if(nwalkers>=1):
      self.nwalkers = nwalkers
    else:
      raise ValueError("Number of walkers must be a positive integer greater than 1")
    self.std = std 
    if(type(device)==str):
      self.device = torch.device(device)
    else:
      self.device = device
    
    self.chains = torch.empty(self.nwalkers, self.dim, device=self.device, requires_grad=False)   #tensor to store walkers at current step in MCMC chain
    self._init_chains()                                                                           #init the walkers at a random point in the state space.
    
    self.proposal = torch.empty(self.nwalkers, self.dim, device=self.device, requires_grad=False) #prop. dist. to propose the next move in the MCMC chain
    self.uniform = torch.empty(self.nwalkers, device=self.device, requires_grad=False)            #tensor for Metropolis-Hastings accept-reject step
    
    self.acceptance = torch.empty(self.nwalkers, device=self.device, requires_grad=False)         #tensor to store acceptance rate for current sample
    
  @torch.no_grad()
  def _init_chains(self):
    """ init. the walkers """
    nn.init.normal_(self.chains, mean=0., std=1.)
  
  @torch.no_grad()
  def _log_pdf(self, x):
    """ calculate the current log-prob of the walkers """
    return self.network(x)[1].mul(2).detach_()
    
  @torch.no_grad()
  def _sample_x_given_y(self, y):
    """ sample a new position from the current walker position """
    return (self.proposal.normal_(std=self.std) + y).detach_()
    
  @torch.no_grad()
  def step(self):
    xcand = self._sample_x_given_y(self.chains).detach_()                                            #propose a candidate move 
    
    log_u = self.uniform.uniform_().log().detach_()                                                  #calculate uniform log prob (for accept-reject step)
    log_a = self._log_pdf(xcand).detach_() - self._log_pdf(self.chains).detach_()                    #calculate log acceptance probability 
    
    condition = torch.lt(log_u, log_a).int().detach_()                                               #if log_u < log_a accept the move (condition=1 else 0)
    self.acceptance += condition
    
    condition = condition.unsqueeze(1).repeat(1, self.dim).detach_()                                 #reshape condition vector to match walker tensor
    
    self.chains = xcand.mul_(condition).detach_() + self.chains.mul_(1-condition).detach_()          #update the accepted moves, else keep the same value
    
  @torch.no_grad()
  def forward(self, nsamples, burn_in, thinning):
    values = torch.zeros(self.nwalkers, nsamples, self.dim, device=self.device, requires_grad=False) #tensor to store samples
    self.acceptance.zero_()                                                                          #zero the acceptance tensor,
                                                                                                     #so we don't count acceptance from previous call
    total_samples = thinning*(nsamples+1) + burn_in                                                  #total number of samples
    
    for i in range(total_samples):                                                                   #cycle through the samples 
      self.step() #step the chain
      if(i>burn_in and i%thinning==0): 
        idx=(i-burn_in)//thinning - 1
        values[:,idx,:] = self.chains.detach_()                                                      #copy current chain value into the samples tensor
    return values.detach_(), (100.0*self.acceptance.detach_()/total_samples)                         #return values (and acceptance rates)

