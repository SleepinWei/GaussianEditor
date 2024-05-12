import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def plot_grad_flow_gaussian(gaussian):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    named_parameters = {
        "xyz": gaussian.get_xyz,
        "features_dc": gaussian.get_features_dc,
        "features_rest": gaussian.get_features_rest,
        "scaling": gaussian.get_orig_scaling,
        "rotation": gaussian.get_orig_rotation,
        "opacity": gaussian.get_orig_opacity,
    }

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters.items():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
            max_grads.append(p.grad.abs().max().cpu().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("./vis_temp/gradient.jpg")

import torch
def plot_histogram_gaussian(gaussian,path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    named_parameters = {
        "xyz": gaussian.get_xyz,
        "features_dc": gaussian.get_features_dc,
        "features_rest": gaussian.get_features_rest,
        "scaling": gaussian.get_orig_scaling,
        "rotation": gaussian.get_orig_rotation,
        "opacity": gaussian.get_orig_opacity,
    }

    ave_grads = []
    max_grads= []
    layers = []
    for index, (n, p) in enumerate(named_parameters.items()):
        if(p.requires_grad) and ("bias" not in n):
            plt.subplot(3,2,index+1)
            plt.cla()
            grad = p.grad[torch.abs(p.grad) > 0.001]
            if grad.ndim == 3:
                # plt.hist(grad[:,0,:].cpu().numpy(),bins=20,edgecolor="black",density=True)
                grad = grad[:,0,:].cpu().numpy()
            else:
                grad = grad.cpu().numpy()
                # plt.hist(grad.cpu().numpy(),bins=20,edgecolor="black",density=True)
            counts, bins = np.histogram(grad,bins=40)
            # counts = counts / p.grad.shape[0]
            plt.stairs(counts,bins,fill=True)
            plt.title(n)
    plt.savefig(path) # "./vis_temp/gradient_histogram.jpg")

