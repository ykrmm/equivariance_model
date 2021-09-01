from torch import nn
import torch.nn.functional as F



class JSD(nn.Module):
    
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=1)
        net_2_probs=  F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_1_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction="batchmean") 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction="batchmean") 
     
        return (0.5 * loss)