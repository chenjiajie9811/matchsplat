from dataclasses import dataclass
from jaxtyping import Float

import torch
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss




'''
Pose Loss
'''
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))*-1 )
    
    theta = torch.acos(cos)

    return theta.mean()

@dataclass
class LossPoseCfg:
    weight: float = 1.0
    

@dataclass
class LossPoseCfgWrapper:
    pose: LossPoseCfg

class LossPose(Loss[LossPoseCfg, LossPoseCfgWrapper]):
    def forward(
            self,
            prediction: DecoderOutput,
            batch: BatchedExample,
            gaussians: Gaussians,
            global_step: int,
        ) -> Float[Tensor, ""]:

        return self.cfg.weight * (
            (compute_geodesic_distance_from_two_matrices(
                batch['rel_pose'][:,:3,:3], 
                batch['gt_rel_pose'][:,:3,:3])) + 
            torch.norm(batch['rel_pose'][:,:3,3] - batch['gt_rel_pose'][:,:3,3], dim=-1).mean())
        
          
