"""
     ---> model_T1 ---> logits == logits <------
    |                                           |
input --> model_S -----------------------> features ---> logits == labels 

"""

import os
import time
import json
import argparse
import copy

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from  torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import prompters
from engine import train_kd, test, train_test
from utils import dataloader, network, KD_loss

def parse_option():

    parser = argparse.ArgumentParser('Original distillation Models')

    parser.add_argument('--config', default='config/target.json', help='config file')
    parser.add_argument('--gpu', type=str, default='0', help='gpu to use')
    parser.add_argument('-alpha', nargs='*', default=[], help='distillation alpha')
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--masterport', type=str, default='12345')

    args = parser.parse_args()

    return args

def main():
    alpha_array = [0.8, 0.9]
    torch.multiprocessing.set_start_method('spawn')
    for a in alpha_array:
        main_cycle(a)

def main_cycle(alpha):

    args = parse_option()
    with open(args.config) as config_file:
        state = json.load(config_file)
    
    state['trial'] = args.trial
    state['masterport'] = args.masterport

    distill_folder = 'save/distill/lr_{}_bs_{}_ep_{}_trial_{}'.format(state['learning_rate_fc'], state['batch_size'], state['epochs'], state['trial'])
    if not os.path.isdir(distill_folder):
        os.makedirs(distill_folder)

   # environment settings
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_gpus = torch.cuda.device_count()

    state['batch_size'] = int(state['batch_size']/num_gpus)

    # launch multiprocessing
    mp.spawn(main_worker, nprocs=num_gpus, args=(state, num_gpus, alpha, distill_folder))

def main_worker(rank, state, num_gpus, alpha, distill_folder):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = state['masterport']
    
    dist.init_process_group(backend='nccl',
                            init_method=state["dist_url"],
                            world_size=num_gpus,
                            rank=rank)
                            
    torch.cuda.set_device(rank)
    torch.distributed.barrier()

    # data preparation
    if rank == 0:
        print('==> Preparing data..')

    train_loader, train_sampler, _ = dataloader('train', rank, num_gpus, state)

    val_loader, val_sampler, classes = dataloader('test', rank, num_gpus, state)

    torch.distributed.barrier()

    # create model
    if rank == 0:
        print('==> Building model..')
    
    model = network(state["net"], classes, rank)

    # teacher model and corresponding distillation alpha preparation
    modelT = network(state["netT_type"], classes, rank, netname=state["netT_name"])

    for param in modelT.parameters():
        param.requires_grad = False
        
    prompter = prompters.__dict__[state["method"]](rank, state).to('cuda')
    prompter = DDP(prompter, device_ids=[rank])
    _ = prompter.module.pad.shape

    # prompter.load_state_dict(torch.load(os.path.join('save/teacher', state["initialization"]), map_location=torch.device('cpu')))

    # data augmentation
    mixup = 0.8
    cutmix = 1.0
    cutmix_minmax = None
    mixup_prob = 1.0
    mixup_switch_prob = 0.5
    mixup_mode = 'batch'
    smoothing = 0.1

    mixup_fn = None
    mixup_active = mixup > 0 or cutmix > 0. or cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
            prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
            label_smoothing=smoothing, num_classes=classes)

    # criterion
    if mixup_active:
        criterion_train = SoftTargetCrossEntropy().cuda(rank)
    elif smoothing:
        criterion_train = LabelSmoothingCrossEntropy(smoothing).cuda(rank)
    else:
        criterion_train = torch.nn.CrossEntropyLoss().cuda(rank)

    criterion_test = torch.nn.CrossEntropyLoss().cuda(rank)

    warmup = len(train_loader) * state["warmup_epoch"]

    # fc model and optimizer & scheduler
    KD_LOSS = KD_loss(model.module.fc.in_features, int(state["netT_classes"])).cuda(rank)
    
    class opt_sch():
        def __init__(self, model, KD_LOSS):
            # optimizer
            self.optimizer = torch.optim.SGD(model.parameters(), lr=state['learning_rate_fc'], momentum=state['momentum'], weight_decay=state['weight_decay'])
            # scheduler
            self.scheduler = CosineLRScheduler(self.optimizer, t_initial=len(train_loader)*state['epochs'], warmup_t=warmup)
            self.Optimizer = torch.optim.SGD(KD_LOSS.parameters(), lr=state['learning_rate_fc'], momentum=state['momentum'], weight_decay=state['weight_decay'])
            self.Scheduler = CosineLRScheduler(self.Optimizer, t_initial=len(train_loader)*state['epochs'], warmup_t=warmup)

    # tensorboard writer
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join('runs', time.strftime(f"%Y-%m-%d {time.localtime().tm_hour+8}:%M:%S", time.localtime()), ' - ', distill_folder.split('/')[-1]))

    # training loop
    best_acc = 0.0
    for epoch in range(state['epochs']):
        
        if rank == 0:
            print('\nEpoch: %d' % (epoch+1))

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
    
        KD_LOSSES = []
        Model = []
        Noise = torch.empty(0, _[0], _[1], _[2], _[3]).cuda()
        Loss = []
        Acc_train = []
        
        for i in range(state["samples"]):
            
            model_p = copy.deepcopy(model)
            KD_LOSS_p = copy.deepcopy(KD_LOSS)
            prompter_p = copy.deepcopy(prompter)
            for param in prompter_p.parameters():
                param.requires_grad = False
            noise = prompter_p.module.perturbations(state['sigma'], noise=noise if i%2 != 0 else None)
            
            opt = opt_sch(model_p, KD_LOSS_p)
            # train model
            acc_train_pre = train_kd(model_p, modelT, prompter_p, train_loader, mixup_fn, criterion_train, opt.optimizer, opt.scheduler, KD_LOSS_p, opt.Scheduler, opt.Optimizer, rank, num_gpus, alpha, state, epoch)
                    
            # test model
            with torch.no_grad():
                acc_train, loss_test = train_test(model_p, train_loader, mixup_fn, criterion_train, rank, num_gpus)
                
            Model.append(model_p)
            KD_LOSSES.append(KD_LOSS_p)
            Loss.append(loss_test)
            Noise = torch.cat((Noise, noise.unsqueeze(0)), dim=0)
            Acc_train.append(acc_train)
        
        Loss_T = torch.tensor(Loss).cuda()

        Grad = torch.mean((Loss_T.expand_as(Noise.transpose(0,4)) * Noise.transpose(0,4)).transpose(0,4), dim=0)
        
        lr = state["lr"] * np.cos((np.pi/2) * epoch/state["epochs"])
        prompter.module.pad.data -= torch.nn.Parameter(lr * Grad)
        
        i = Acc_train.index(max(Acc_train))
        model = Model[i]
        KD_LOSS = KD_LOSSES[i]
        # test model
        with torch.no_grad():
            acc_test = test(model_p, val_loader, criterion_test, rank, num_gpus)
            
        acc_train = Acc_train[i]

        # save model
        if rank == 0:
            acc = acc_test
            if best_acc < acc:
                filename_sub = 'alpha_{alpha}_acc:{best_acc}.pth'.format(alpha='%.1f' % alpha, best_acc=format(best_acc, '.6f'))
                filename_best = 'alpha_{alpha}_acc:{acc}.pth'.format(alpha='%.1f' % alpha, acc=format(acc, '.6f'))
                sub_path = os.path.join(distill_folder, filename_sub)
                best_path = os.path.join(distill_folder, filename_best)

                if best_acc != 0:
                        os.remove(sub_path)
                torch.save(model.state_dict(), best_path)
                best_acc = acc
            
            writer.add_scalar('Train/Accuracy', acc_train , epoch)
            writer.add_scalar('Test/Accuracy', acc_test , epoch)
            writer.add_scalar('Padsum', prompter.module.padsum() , epoch)
            writer.add_scalar('Pad_lr', lr, epoch)
            writer.add_scalar('Grad', torch.mean(Grad), epoch)
            writer.add_scalar('test_train', torch.mean(Loss_T), epoch)

    if rank == 0:
        writer.close()
    torch.distributed.barrier()
    print("Done!")

if __name__ == '__main__':
    main()