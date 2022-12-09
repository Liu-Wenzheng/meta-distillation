"""
     ---> model_T1 ---> logits == logits <------
    |                                           |
input --> model_S -----------------------> features ---> logits == labels 

"""

import os
import time
import json
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from  torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import prompters
from engine import train_rl, test
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

    prompter = prompters.__dict__[state["method"]](state).to('cuda')
    prompter = DDP(prompter, device_ids=[rank])

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

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=state['learning_rate_fc'], momentum=state['momentum'], weight_decay=state['weight_decay'])
    warmup = len(train_loader) * state["warmup_epoch"]
    # scheduler
    scheduler = CosineLRScheduler(optimizer, t_initial=len(train_loader)*state['epochs'], warmup_t=warmup)

    # fc model and optimizer & scheduler
    KD_LOSS = KD_loss(model.module.fc.in_features, int(state["netT_classes"])).cuda(rank)
    Optimizer = torch.optim.SGD(KD_LOSS.parameters(), lr=state['learning_rate_fc'], momentum=state['momentum'], weight_decay=state['weight_decay'])
    Scheduler = CosineLRScheduler(Optimizer, t_initial=len(train_loader)*state['epochs'], warmup_t=warmup)

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

        # train model
        acc_train = train_rl(model, modelT, prompter, train_loader, val_loader, mixup_fn, criterion_train, criterion_test, optimizer, scheduler, KD_LOSS, Scheduler, Optimizer, rank, num_gpus, alpha, state, epoch)
        
        raise Exception('none')
        # test model
        with torch.no_grad():
            acc_test = test(model, val_loader, criterion_test, rank, num_gpus)

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

    if rank == 0:
        writer.close()
    torch.distributed.barrier()
    print("Done!")

if __name__ == '__main__':
    main()