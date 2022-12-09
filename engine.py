import torch
from utils import progress_bar, fc_input
import torch.distributed as dist

from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt/nprocs

def test(model, val_loader, criterion, rank, num_gpus):
    model.eval()

    test_loss = 0.0
    correct = 0.0
    total = 0.0
    length = 0

    for batch_index, (images, labels) in enumerate(val_loader):

        images = images.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)

        # forward propogation
        outputs = model(images)
        loss = criterion(outputs, labels)

        torch.distributed.barrier()

        # result statistics
        test_loss += reduce_mean(loss, num_gpus).item()
        total += labels.size(0)
        _, preds = outputs.max(1)
        correct += reduce_mean(preds.eq(labels).sum(), num_gpus).item()

        if rank == 0:
            length = progress_bar(batch_index, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/((batch_index+1)), 100.*correct/total, correct, total), last_length=length)
    
    return correct/total

def train_kd(model, modelT, prompter, train_loader, mixup_fn, criterion, optimizer, scheduler, KD_LOSS, Scheduler, Optimizer, rank, num_gpus, alpha, state, epoch):

    model.train()
    prompter.eval()

    train_loss = 0.0
    correct = 0.0
    total = 0.0
    length = 0

    num_batches_per_epoch = len(train_loader)
    print(f"rank: {rank}, padsum: {prompter.module.padsum()}")

    for batch_index, (images, labels) in enumerate(train_loader):
        # adjust learning rate
        step = num_batches_per_epoch * epoch + batch_index
        scheduler.step(step)
        Scheduler.step(step)

        images = images.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)

        # data augmentation
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # add hooks to student model
        FC = fc_input(model)
        FC.fc_input_data()

        # student model forward propogation
        pred = model(images.requires_grad_())

        # extract pre_fc layer features
        Feature = FC.fc_input()

        # teacher models forward propogation
        with torch.no_grad():
            images = resize(images, 224, InterpolationMode.BICUBIC)
            imaged = prompter(images)
            predT = modelT(imaged)
        
        # distillation related loss calculation
        loss_kd = KD_LOSS.forward(Feature, predT, state['T'])

        if mixup_fn is not None:
            L = torch.nn.LogSoftmax(dim=1)
            loss_l = criterion(L(pred), labels)
        else:
            loss_l = criterion(pred, labels)

        loss = loss_l * (1 - alpha) + loss_kd * alpha

        torch.distributed.barrier()

        # student and teacher models backward propogation
        optimizer.zero_grad()
        Optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        Optimizer.step()

        # result statistics
        train_loss += reduce_mean(loss, num_gpus).item()
        total += labels.size(0)
        _, predicted = pred.max(1)
        if mixup_fn is not None:
            _, target = labels.max(1)
            correct += reduce_mean(predicted.eq(target).sum(), num_gpus).item()
        else:
            correct += reduce_mean(predicted.eq(labels).sum(), num_gpus).item()
            
        if rank == 0:
            length = progress_bar(batch_index, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_index+1), 100.*correct/total, correct, total), last_length=length)

    return correct/total

def train_test(model, train_loader, mixup_fn, criterion, rank, num_gpus):

    model.eval()

    train_loss = 0.0
    correct = 0.0
    total = 0.0
    length = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)

        # data augmentation
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # student model forward propogation
        pred = model(images.requires_grad_())

        if mixup_fn is not None:
            L = torch.nn.LogSoftmax(dim=1)
            loss = criterion(L(pred), labels)
        else:
            loss = criterion(pred, labels)

        # result statistics
        train_loss += reduce_mean(loss, num_gpus).item()
        total += labels.size(0)
        _, predicted = pred.max(1)
        if mixup_fn is not None:
            _, target = labels.max(1)
            correct += reduce_mean(predicted.eq(target).sum(), num_gpus).item()
        else:
            correct += reduce_mean(predicted.eq(labels).sum(), num_gpus).item()
            
        if rank == 0:
            length = progress_bar(batch_index, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_index+1), 100.*correct/total, correct, total), last_length=length)

    return correct/total, train_loss/(batch_index+1)
