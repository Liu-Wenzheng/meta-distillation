import sys
import time
import os
import random
from torchvision import datasets, transforms

import torch
import torch.nn as nn
from  torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler as DS


def network(nettype, classes, local_rank, netname='none'):

    if nettype == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif nettype == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif nettype == 'ViT-B/16':
        from timm.models import create_model
        net = create_model(
        "deit_base_patch16_224",
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if nettype != 'ViT-B/16':
        if net.fc.out_features != classes:
            net.fc = nn.Linear(net.fc.in_features, classes)
            
    if netname != 'none':
        net.load_state_dict(torch.load(os.path.join('save/teacher', netname), map_location=torch.device("cpu")))
	    
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda(local_rank)
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=False)
    cudnn.benchmark = True

    return net

def datasetslice(dataset, split, num=0, shuffle=True):
    if num == 0:
        split_indices=[]
        sequence=[0]
        k = 0
        for i in split:
            split_indices.append([])
            k+=i
            sequence.append(k)
        for i in range(len(dataset)):
            current_class = dataset[i][1]
            for j in range(len(split)):
                if sequence[j] <= current_class < sequence[j+1]:
                    split_indices[j].append(i)
    else:
        Sum = sum(split)
        indices = []
        for i in range(Sum):
            indices.append([])
        for i in range(len(dataset)):
            indices[dataset[i][1]].append(i)
        if shuffle == True:
            for i in range(Sum):
                random.shuffle(indices[i])
        split_indices=[]
        sequence=[0]
        k = 0
        for i in split:
            split_indices.append([])
            k+=i
            sequence.append(k)
        for current_class in range(Sum):
            for i in range(len(split)):
                if sequence[i] <= current_class < sequence[i+1]:
                    split_indices[i] += indices[current_class][:num]

    return split_indices

def dataset(setname):
    if setname == 'cifar100':
        Set = datasets.CIFAR100
        out_classes = 100
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif setname == 'cifar10':
        Set = datasets.CIFAR10
        out_classes = 10
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise Exception("Invalid setname.")

    return Set, out_classes, mean, std

def dataloader(type, rank, num_gpus, state):
    
    Set, out_classes, mean, std = dataset(state['dataset'])
    
    if type == 'train':
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        datasets = Set(root='./data', train=True, download=True, transform=transform_train)
    elif type == 'test':
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        datasets = Set(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise Exception("type should be train or test.")
    
    sampler = DS(datasets, num_replicas=num_gpus, rank=rank)
    loader = DataLoader(datasets, num_workers=state['num_workers'], batch_size=state['batch_size'], sampler=sampler, pin_memory=True, persistent_workers=True)
    
    return loader, sampler, out_classes

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

TOTAL_BAR_LENGTH = 20.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None, last_length=0):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    L.append(' %d/%d ' % (current+1, total))

    msg = ''.join(L)
    sys.stdout.write(msg)
    
    length = TOTAL_BAR_LENGTH + 3 + len(msg)
    complement = int(last_length - length)
    if complement > 0:
        sys.stdout.write(complement * ' ')
    
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    return length

class KD_loss(nn.Module):
    def __init__(self, s_dim, t_dim):
        super(KD_loss, self).__init__()
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.fc = nn.Linear(self.s_dim, self.t_dim)

    def forward(self, output_f, t_logits, T):
        s_logits = self.fc(output_f)
        L = nn.LogSoftmax(dim=1)
        criterionT = nn.KLDivLoss(reduction="batchmean", log_target=True)
        return criterionT(L(t_logits/T), L(s_logits/T))*(T*T)
    

class fc_input():

    def __init__(self, model):
        self.model = model
        self.input_m = []
        self.hook = None

    def fc_input_data(self):
        
        def hook_fn_forward(module, input, output):
            self.input_m.append(input)

        for name, module in self.model.module.named_children():
            if name == 'fc':
                self.hook = module.register_forward_hook(hook_fn_forward)

        if self.hook == None:
            raise Exception("No fc layer found!")

    def fc_input(self):
        result, = self.input_m[0]
        self.hook.remove()
        return result

def datainput(netT_name, netT_type, netT_classes, T, alpha):

    num = len(netT_name)
    if num != len(alpha):
        raise Exception("netT_name and alpha dimensions don't match")
    else:
        alpha = [float(i) for i in alpha]

    if len(netT_type) == 1:
        netT_type = [netT_type[0] for i in range(num)]
    elif len(netT_type) != num:
        raise Exception("number of netT_type Error")

    if len(T) == 1:
        T = [float(T[0]) for i in range(num)]
    elif len(T) == num:
        T = [float(i) for i in T]
    else:
        raise Exception("number of T Error")

    if len(netT_classes) == 1:
        netT_classes = [int(netT_classes[0]) for i in range(num)]
    elif len(netT_classes) == num:
        netT_classes = [int(i) for i in netT_classes]
    else:
        raise Exception("number of netT_classes Error")

    return num, netT_name, netT_type, netT_classes, T, alpha

        

