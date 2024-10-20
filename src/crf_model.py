import torch
from torch import nn
from crfseg.model import CRF
from dataclasses import dataclass


class CRFAugmented(nn.Module):
    def __init__(self, net):
        super(CRFAugmented, self).__init__()
        self.net = net
        self.crf = CRF(n_spatial_dims=2)

    def forward(self, emb, multimask_output=True, image_size=512):
        # Unpack inputs and pass them to the net
        net_output = self.net(emb, multimask_output, image_size)
        # If net_output is a tuple, the CRF should only take the final output (adjust as needed)
        if isinstance(net_output, tuple):
            net_output = net_output[0]
        # Pass the output of the net into the CRF
        crf_output = self.crf(net_output["masks"])
        return {"masks": crf_output}


def apply_crf(net, args):
    for param in net.parameters():
        param.requires_grad = False

    if args.finetune_crf:
        # Freeze the provided model except the last layer
        layers = list(net.children())

        # Now unfreeze the last layer
        for param in layers[-1].parameters():
            param.requires_grad = True

    if args.model == "samed" or args.model == "samed_fast":
        model = CRFAugmented(net)
    else:
        # Add the CRF layer to the model
        model = nn.Sequential(net, CRF(n_spatial_dims=2))
    return model
