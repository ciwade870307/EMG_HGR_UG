import numpy as np 
import matplotlib.pyplot as plt

# Torch
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

# Progress Bar
from rich.progress import track
from rich.progress import Progress
from rich.console import Console
import sys

# Other
import time

# train(...., model_type='ViT')

# def validate(model, loss_fn, dataloader, device):
#     model.eval()
#     model.to(device)
#     for x, y in dataloader:
#         with torch.no_grad():
#             y_pred
#     return loss, acc

# def train_epoch(model, optimizer, loss_fn, dataloader, device, **kwargs):
#     model.train()
#     model.to(device)
#     for x, y in dataloader:
#         y_pred
#         loss = loss_fn(y_pred)

#         opt...

#     return loss, acc

# config = {
#     'model_type': 'ViT',
#     'window_size': 16,
# }

# config = {
#     'model_type': 'CNN',
#     'depth': 16
# }

# train(config_ViT)

# def train(config):
#     pass

# train(model_ViT,)

# # config: dict
# def train(model, optimizer, scheduler, loss_fn, train_dataloader, val_dataloader, checkpoint_path, epochs, device, config, **kwargs):
#     if 'model_type' in kwargs:
#         if kwargs['model_type'] == 'ViT':
#             pass
    
#     for epoch in range(1, epochs+1):
#         print()
#         train_loss, train_acc = train(model, optimizer, loss_fn, train_dataloader, device, **kwargs)
#         val_loss, val_acc

#         if val_loss < best:
#             pass

#         print('[Epoch: %d] train-loss: %6.4f, ')

#     # return val_acc

def train_process(args,model,model_PATH,train_loader,valid_loader,device,optimizer,criterion,scheduler=None):
    # Print the model summary
    # summary(model, (train_loader.dataset[0][0].shape))
    
    # model.load_state_dict(torch.load(model_PATH)) # Pretrained model

    t_start = time.time()
    # with Progress(console=Console(file=sys.stderr)) as progress:
    #with Progress() as progress:
    # task1 = progress.add_task("[cyan]Epoch: ", total=args.num_epoch)

    loss_best = float("inf")
    not_better_count = 0    
    # print('', end='', flush=True)
    for epoch in range(args.num_epoch):
        # end_type = '\n' if epoch==args.num_epoch-1 else '\r'
        end_type = '\n'
        
        # Training Stage
        model.train() # Make sure the model is in train mode before training.
        # task2 = progress.add_task("[green][Train: ]", total=len(train_loader))
        running_loss_train = 0.0
        running_acc_train = 0.0
        n_train = 0
        for i, (emg_sample, gesture_gold) in enumerate(train_loader):
            emg_sample = emg_sample.to(device) # input
            gesture_gold = gesture_gold.to(device) # Output

            if args.model_type == "ViT":
                # emg_sample = torch.unsqueeze(emg_sample.permute(0,2,1), dim=3)  # shape: (B, C, W, F) = (128, 12, 400, 1)
                emg_sample = emg_sample.permute(0,2,1)  # shape: (B, C, 1*W*F) 

            gesture_pred = model(emg_sample)
            
            optimizer.zero_grad()
            loss = criterion(gesture_pred,gesture_gold)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.detach().item()*emg_sample.shape[0]
            num_correct = (torch.argmax(gesture_pred, dim=1)==gesture_gold).sum()
            running_acc_train += num_correct
            n_train += emg_sample.shape[0]
            # progress.update(task2, advance=1,description="[green][Train: %3d] [Train loss: %3.3f] [Train acc: %3.2f %%]: " % (i+1,running_loss_train/n_train, 100*running_acc_train/n_train))
            
        # print("[Train: %3d] [Train loss: %3.3f] [Train acc: %3.2f %%]: " % (i+1,running_loss_train/n_train, 100*running_acc_train/n_train), end=end_type)
        # progress.remove_task(task2)

        # Validation Stage
        model.eval() # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        # task3 = progress.add_task("[red]Validate ", total=len(valid_loader))
        running_loss_valid = 0.0
        running_acc_valid = 0.0
        n_valid = 0
        with torch.no_grad():
            for i, (emg_sample, gesture_gold) in enumerate(valid_loader):
                emg_sample = emg_sample.to(device) # input

                if args.model_type == "ViT":
                    # emg_sample = torch.unsqueeze(emg_sample.permute(0,2,1), dim=3)  # shape: (B, C, W, F) = (128, 12, 400, 1)
                    emg_sample = emg_sample.permute(0,2,1)  # shape: (B, C, 1*W*F)

                gesture_gold = gesture_gold.to(device) # Output
                gesture_pred = model(emg_sample)

                loss = criterion(gesture_pred,gesture_gold)

                running_loss_valid += loss.detach().item()*emg_sample.shape[0]
                num_correct = (torch.argmax(gesture_pred, dim=1)==gesture_gold).sum() # torch.mean((torch.argmax(gesture_pred, dim=1) == labels).float()
                

                running_acc_valid += num_correct
                n_valid += emg_sample.shape[0]
                # progress.update(task3, advance=1,description="[red][Validate: %3d] [Valid loss: %3.3f] [Valid acc: %3.2f %%]: " % (i+1, running_loss_valid/n_valid, 100*running_acc_valid/n_valid))
            # print("[Valid: %3d] [Valid loss: %3.3f] [Valid acc: %3.2f %%]: " % (i+1, running_loss_valid/n_valid, 100*running_acc_valid/n_valid),  end=end_type)
        # progress.remove_task(task3)
        
        # Finishing a epoch
        # progress.update(task1, advance=1,description="[cyan][Epoch: %3d] [Train loss: %3.3f acc: %3.2f %%] [Valid loss: %3.3f acc: %3.2f %%]" 
        #                 %(epoch+1, running_loss_train/n_train, 100*running_acc_train/n_train, running_loss_valid/n_valid, 100*running_acc_valid/n_valid))
        print("[Epoch: %3d / %d] [Train loss: %3.3f acc: %4.2f] [Valid loss: %3.3f acc: %4.2f]" 
                        %(epoch+1, args.num_epoch, running_loss_train/n_train, 100*running_acc_train/n_train, running_loss_valid/n_valid, 100*running_acc_valid/n_valid),  end=end_type, flush=True)

        # Save model or not 
        if running_loss_valid < loss_best:
            loss_best = running_loss_valid
            # print("New best loss in valid => Saving the model @ %s", PATH)
            torch.save(model.state_dict(), model_PATH)
            not_better_count = 0
        else:
            not_better_count = not_better_count+1
        if not_better_count > 500 and epoch>30:
            break
        if scheduler is not None:
            scheduler.step()
    
    t_end = time.time()
    print("Elasped training time: ", t_end - t_start)    
                

def test_process(model,model_PATH,test_loader,device,criterion,load_model,model_type):
    if load_model == True:
        print("model_PATH: ",model_PATH)
        model.load_state_dict(torch.load(model_PATH))
        

    # with Progress() as progress:
    # Testing Stage
    model.eval() # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    # task4 = progress.add_task("[blue]Batch ", total=len(test_loader))
    running_loss_test = 0.0
    running_acc_test = 0.0
    n_test = 0

    y_pred = []
    y_gold = []

    with torch.no_grad():
        for i, (emg_sample, gesture_gold) in enumerate(test_loader):
            emg_sample = emg_sample.to(device) # input
            gesture_gold = gesture_gold.to(device) # Output

            if model_type == "ViT":
                # emg_sample = torch.unsqueeze(emg_sample.permute(0,2,1), dim=3)  # shape: (B, C, W, F) = (128, 12, 400, 1)
                emg_sample = emg_sample.permute(0,2,1)  # shape: (B, C, 1*W*F)
            gesture_pred = model(emg_sample)

            y_pred.extend(gesture_pred.argmax(dim=-1).view(-1).detach().cpu().numpy())       # 將preds預測結果detach出來，並轉成numpy格式       
            y_gold.extend(gesture_gold.view(-1).detach().cpu().numpy())      # target是ground-truth的labe
            
            loss = criterion(gesture_pred,gesture_gold)

            running_loss_test += loss.detach().item()*emg_sample.shape[0]
            num_correct = (torch.argmax(gesture_pred, dim=1)==gesture_gold).sum()
            running_acc_test += num_correct
            n_test += emg_sample.shape[0]

            # progress.update(task4, advance=1,description="[blue][Test loss: %3.3f] [Test acc: %3.2f %%]: " % (running_loss_test/n_test, 100*running_acc_test/n_test))
        print("[Test loss: %3.3f] [Test acc: %3.2f %%] " % (running_loss_test/n_test, 100*running_acc_test/n_test), flush=True)

    # Print the model summary
    # summary(model, (emg_sample.shape[1:]))

    # return y_pred, y_gold
