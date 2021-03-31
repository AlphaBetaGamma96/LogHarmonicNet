import torch
import torch.nn as nn

class HarmonicOscillator1D(nn.Module):

  def __init__(self, network):
    super(HarmonicOscillator1D, self).__init__()
    
    self.network = network #network instance
    
    #some physical constants...
    self.mass = 939 #MeV 
    self.hbar = 197 #MeV fm
    self.hbar_omega = 40*self.network.num_input**(-1/3) #MeV
    self.omega = self.hbar_omega/self.hbar #fm^-1
    
    self.kinetic_const = -(self.hbar**2)/(2*self.mass) #MeV fm^2
    self.potential_const = 0.5*self.mass*self.omega**2 #MeV fm^-2
      
  def kinetic_from_log_per_walker(self, xs):
    """
    Method to calcualte the local kinetic energy values of a netork function, f, and samples, x, of a single 
    walker/chain.
    The values calculated here are 1/f d2f/dx2 which is equivalent to d2log(|f|)/dx2 + (dlog(|f|)/dx)^2
    within the log-domain (rather than the linear-domain).
    """
    xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
    xs_flat = torch.stack(xis, dim=1)
    
    _, ys = self.network(xs_flat.view_as(xs))
    
    ones = torch.ones_like(ys)
    (dy_dxs, ) = torch.autograd.grad(ys, xs_flat, ones, retain_graph=True, create_graph=True)
    
    lay_ys = sum(torch.autograd.grad(dy_dxi, xi, ones, retain_graph=True, create_graph=False)[0] \
                  for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))               
    )
    
    ek_local_per_walker = self.kinetic_const * (lay_ys + dy_dxs.pow(2).sum(-1)) #move const out of loop?
    return ek_local_per_walker.detach()
    
  def local_kinetic_from_log(self, x):
    """
    Method to calculate the total kinetic local energy for all walkers by calculating for a single walker and stack
    all walkers together.
    """ 
    return torch.stack([self.kinetic_from_log_per_walker(x[walker, :, :]) for walker in range(x.shape[0])],dim=0)
    
  def local_potential(self, x):
    """
    Method to calculate the total potential local energy for all walkers. We can calculate this in parallel as its 
    just the squared sum along dim=2 (position dimension).
    """
    ep_locals = self.potential_const*(x.pow(2).sum(dim=-1))
    return ep_locals
    
  def forward(self, x):
    ek_local = self.local_kinetic_from_log(x)
    ep_local = self.local_potential(x)
    e_local = ek_local + ep_local
    return e_local
    
