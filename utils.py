import torch
import psutil, os

def memory_usage(prefix):
  if(prefix=='GB'):
    base=2.**30
  elif(prefix=='MB'):
    base=2.**20
  elif(prefix=='KB'):
    base=2.**10
  ram=psutil.Process(os.getpid()).memory_info()[0]/base
  return float(ram)
  
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
  
def calc_clip(E_local, clip_factor):
  median=torch.median(E_local)
  variation=torch.mean( torch.abs(E_local - torch.median(E_local)) )
  return median + clip_factor*variation
  
def pretraining_loss_all(X, net_psi, targ_psi, nblocks):
  nwalkers=X.shape[0]
  
  walker_mean = torch.empty(nwalkers, device=X.device)
  walker_var = torch.empty(nwalkers, device=X.device)
  walker_Tc = torch.empty(nwalkers, device=X.device)
  
  for i, walker in enumerate(range(nwalkers)):
    net_slater = net_psi(X[i,:,:])
    targ_slater = targ_psi(X[i,:,:])
    
    local_loss = 0.5*torch.sum( (net_slater - targ_slater)**2, dim=(-2,1))
    
    chain_mean = torch.mean(local_loss)
    chain_var = torch.mean( (local_loss - chain_mean)**2 )
    
    loss_blocks = torch.chunk(local_loss, nblocks)
    block_mean = torch.stack([block.mean() for block in loss_blocks])
    block_var = torch.mean( (block_mean - chain_mean)**2 )
    
    nsteps = X.shape[1]/nblocks #was 0
    
    walker_mean[i] = chain_mean
    walker_var[i] = block_var
    walker_Tc[i] = nsteps*block_var/chain_var
    
  weight = 1.0/walker_var
  loss_mean = torch.sum(weight*walker_mean)/torch.sum(weight)
  loss_var = torch.sum(weight**2*walker_var)/torch.sum(weight**2)
  
  return loss_mean, loss_var
