import torch
import torch.nn as nn

from Layers import InputEquivariantLayer, IntermediateEquivariantLayer
from Layers import SLogSlaterDeterminant

class LogHarmonicNet(nn.Module):

  def __init__(self, num_input, num_hidden, num_layers, num_dets, func):
    super(LogHarmonicNet, self).__init__()
    """
    Permutational Equivariant Neural Network which takes the one-dimensional positions
    of the system (represented by a vector) and returns the log. abs. determinant
    (and its sign).
    """
    self.num_input=num_input
    self.num_hidden=num_hidden
    self.num_layers=num_layers
    self.num_dets=num_dets
    self.func=func
    
    layers = []
    self.input_layer = InputEquivariantLayer(in_features=2,
                                             out_features=self.num_hidden,
                                             num_particles=self.num_input,
                                             func=func,
                                             bias=True)
    for i in range(1, self.num_layers):
      layers.append(IntermediateEquivariantLayer(in_features=2*self.num_hidden,
                                             out_features=self.num_hidden,
                                             num_particles=self.num_input,
                                             func=func,
                                             bias=True)
                   )

    self.layers = nn.ModuleList(layers)
    self.slater = SLogSlaterDeterminant(in_features=self.num_hidden,
                                        num_particles=self.num_input,
                                        bias=True)
                                        
    self.width = nn.Parameter(torch.empty(self.num_dets).fill_(0.1)) #width of envelope 
    
  def forward(self, x0):
    x = self.input_layer(x0)
    for l in self.layers:
      x = l(x) + x
    log_envelope = -self.width*x0.pow(2).sum(dim=-1) #to enforce output goes to 0 at large input values.
    sign, logabsdet = self.slater(x)
    return sign, logabsdet + log_envelope
