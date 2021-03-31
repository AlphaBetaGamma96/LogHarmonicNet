import torch
import torch.nn as nn

class ADVMC(nn.Module):
  
  def __init__(self, network):
    super(ADVMC, self).__init__()
    """
    Class to takes the local values from Hamiltonian.py and convert them into a single scalar loss value
    (The expectation value of the energy)
    """
    self.network = network
      
  def calc_log_prob(self, x):
    """
    Calculates the log-prob for all inputs values for all walkers
    """
    return torch.stack([self.network(x[walker, :, :])[1].mul(2) for walker in range(x.shape[0])], dim=0)
    
  def forward(self, x, local_energy, nblocks):
    """
    Calculates the expectation value of the energy for all walkers and samples via the use of an
    Automatic-Differentiable Variational Monte Carlo (ADVMC) estimator [1911.09117]
    """
    log_prob_wf2 = self.calc_log_prob(x)
    log_prob_wf2_detach = log_prob_wf2.detach()
    local_energy = local_energy.detach()
    
    eloc_numerator = torch.exp(log_prob_wf2 - log_prob_wf2_detach)*local_energy
    eloc_denominator = torch.exp(log_prob_wf2 - log_prob_wf2_detach)
    
    #energy mean and variance (including autocorrelation) calculated via the blocking technique as 
    #defined within J. Chem. Phys. 125, 114105 (2006) 
    energy_mean_per_walker_chain = torch.mean(eloc_numerator, keepdim=True, dim=1)/ torch.mean(eloc_denominator, keepdim=True, dim=1) #Eq. 28
    energy_var_per_walker_chain = torch.mean( (eloc_numerator - energy_mean_per_walker_chain)**2, keepdim=True, dim=1) #Eq. 29
    
    nsteps = eloc_numerator.shape[1]/nblocks
    
    eloc_numerator_blocks = torch.chunk(eloc_numerator, nblocks, dim=1)
    energy_mean_per_walker_block = torch.cat([torch.mean(block, keepdim=True, dim=1) for block in eloc_numerator_blocks], dim=1) #Eq. 31
    energy_var_per_walker_block = torch.mean( (energy_mean_per_walker_block - energy_mean_per_walker_chain)**2, keepdim=True, dim=1) #Eq. 30
    
    #Inverse-variance Weighted Averaging of Walkers' means and variance into a single mean/variance     
    weight = 1.0/energy_var_per_walker_block
    energy_mean = torch.sum(weight*energy_mean_per_walker_chain) / torch.sum(weight)
    energy_var = torch.sum(weight**2*energy_var_per_walker_block) / (torch.sum(weight)**2)

    with torch.no_grad():
      sampler_stats = {} #dict to store items of interest
      
      autocorrelation = nsteps*energy_var_per_walker_block/energy_var_per_walker_chain #Eq. 32
      inefficency = nsteps*energy_var_per_walker_block #Eq. 33
      
      sampler_stats['autocorrelation_mean'] = torch.mean(autocorrelation).item()
      sampler_stats['autocorrelation_std'] = torch.std(autocorrelation).item()
      sampler_stats['autocorrelation_min'] = torch.min(autocorrelation).item()
      sampler_stats['autocorrelation_max'] = torch.max(autocorrelation).item()
    
      sampler_stats['inefficency_mean'] = torch.mean(inefficency).item()
      sampler_stats['inefficency_std'] = torch.std(inefficency).item()
      sampler_stats['inefficency_min'] = torch.min(inefficency).item()
      sampler_stats['inefficency_max'] = torch.max(inefficency).item()
    
    return energy_mean, energy_var.sqrt(), sampler_stats
    
