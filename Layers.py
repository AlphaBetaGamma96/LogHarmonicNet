import torch
import torch.nn as nn

class InputEquivariantLayer(nn.Module):

  def __init__(self, in_features, out_features, num_particles, func, bias):
    super(InputEquivariantLayer, self).__init__()
    """
    Input layer which takes in a batch of vectors and returns a batch of matrices 
    representing the input features in a permutationally equivariant manner
    """
    
    self.in_features = in_features
    self.out_features = out_features
    self.num_particles = num_particles
    self.func = func
    self.bias = bias
    
    self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)
    
  def forward(self, h):
    g = h.mean(dim=1, keepdim=True).repeat(1, self.num_particles)
    f = torch.stack((h,g), dim=2)
    return self.func(self.fc(f))

class IntermediateEquivariantLayer(nn.Module):

  def __init__(self, in_features, out_features, num_particles, func, bias):
    super(IntermediateEquivariantLayer, self).__init__()
    """
    Intermediate layer which takes in a batch of matrices and returns a batch of matrices 
    representing the hidden features in a permutationally equivariant manner
    """
    
    self.in_features = in_features
    self.out_features = out_features
    self.num_particles = num_particles
    self.func = func
    self.bias = bias
    
    self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)
    
  def forward(self, h):
    g = h.mean(dim=1, keepdim=True).repeat(1, self.num_particles, 1)
    f = torch.cat((h,g), dim=2)
    return self.func(self.fc(f))
    
class SLogSlaterDeterminant(nn.Module):
  
  def __init__(self, in_features, num_particles, bias):
    super(SLogSlaterDeterminant, self).__init__()
    """
    Final layer of the network which takes a batch of matrices from an intermediate 
    equivariant layer and transforms them into a batch of matrices of NxN where N 
    is the number of inputs to the network. It then takes the signed-log determinant
    of these matrices and returns the log. abs. determinant (along with its sign)
    """
    
    self.in_features = in_features
    self.num_particles = num_particles
    self.bias = bias
    
    self.weight = nn.Parameter(torch.Tensor(self.in_features, self.num_particles))
    if(self.bias is not None):
      self.bias = nn.Parameter(torch.Tensor(self.num_particles))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()
    self.log_factorial = torch.arange(1,self.num_particles+1).log().sum() #normalisation factor of the determinant (within log-domain)
    
  def reset_parameters(self):
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    if(self.bias is not None):
      torch.nn.init.zeros_(self.bias)
    
  def forward(self, h):
    slater_matrix = torch.matmul(h, self.weight) + self.bias
    sign, logabsdet = torch.slogdet(slater_matrix)
    return sign, logabsdet - 0.5*self.log_factorial
    
