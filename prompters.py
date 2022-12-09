import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

class PadPrompter(nn.Module):
    def __init__(self, rank, state):
        super(PadPrompter, self).__init__()

        self.pad_size = state["prompt_size"]
        self.image_size = state["image_size"]
        self.epsilon = state["epsilon"]
        self.rank = rank
        
        self.base_size = self.image_size - self.pad_size*2
        self.pad = nn.Parameter(torch.zeros([1, 3, self.pad_size, 4*(self.image_size-self.pad_size)]))
        
    def perturbations(self, sigma, noise = None):

        if noise == None:
            dimension = list(self.pad.size())
            noise = torch.empty(dimension).cuda()
            nn.init.normal_(noise)
            dist.broadcast(noise, src=0)
            noise = torch.clamp(noise * self.epsilon, -self.epsilon, self.epsilon)
            self.pad += noise * sigma
        else:
            dist.broadcast(noise, src=0)
            self.pad -= noise * sigma
            
        return noise

    def forward(self, x):
        
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()

        self.pad_up = self.pad[:,:,:,:self.image_size]
        self.pad_down = self.pad[:,:,:,self.image_size:2*self.image_size]
        self.pad_left = self.pad[:,:,:,2*self.image_size:3*self.image_size - 2*self.pad_size].transpose(2,3)
        self.pad_right = self.pad[:,:,:,3*self.image_size - 2*self.pad_size:].transpose(2,3)

        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)

        return x + prompt

    def padsum(self):
        return torch.abs(self.pad).sum()


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt

    def padsum(self):
        return torch.abs(self.patch).sum()


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt

    def padsum(self):
        return torch.abs(self.patch).sum()


def padding(rank, state):
    return PadPrompter(rank, state)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)
