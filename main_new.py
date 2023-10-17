# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:40:09 2022

@author: ZJ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math

def get_model_param(args, model, device, train_loader, valid_loader):
    train_model = model.to(device) 
    train_opt = optim.SGD(params = train_model.parameters(),lr = args.lr)
    loss_temp = 0
    for epoch in range(1, args.epochs + 1):
        # Train Model
        valid_loss = 0
        for batch_idx, (train_data, train_target) in enumerate(train_loader): 
            train_opt.zero_grad()
            train_data, train_target = train_data.to(device), train_target.to(device) ###
            train_pred = train_model(train_data)
            train_loss = F.nll_loss(train_pred, train_target)
            train_loss.backward()
            train_opt.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                         epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size,
                         100. * batch_idx / len(train_loader), train_loss.item()))
        acc, valid_loss = test(args, train_model, device, valid_loader)
        
        if epoch == 1:
            loss_temp = valid_loss
            save_param = train_model.parameters()
            continue
        if valid_loss > loss_temp:
            print('----------------------Return Train Epoch: {} ------------------------'.format(epoch))
            return save_param
        save_param = train_model.parameters() 
        loss_temp = valid_loss
#     print('-------------------Train Epoch Is Not Enough-------------------')
    return save_param

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset),test_loss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# Preprocess data and train models
class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 5000
        self.lr = 0.0001       
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1024
        self.log_interval = 30
        self.save_model = False

args = Arguments()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__=='__main__':
    # Set random seed and CUDA
    setup_seed(5)
    use_cuda = torch.cuda.is_available()
    print('------  use_cuda:' + str(use_cuda) +'----------')
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Loading MNIST data
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

    train_dataset_total = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
  
    # Split training set and valid set
    train_dataset,valid_dataset = train_test_split(train_dataset_total,test_size=0.2,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    indices = [i for i in range(len(train_dataset))]
    random.shuffle(indices)
    
    # Randomly select 500 pieces of training set
    alice_indices = indices[30000:30500]     
    
    
    initial_model = Net() #initial_model = torch.load('./model.pkl')#
    
    results = pd.DataFrame(columns = ['data_size', 'price', 'income1', 'income2'])
    
    # Change the value of c0 to obtain the experimental results(we examine 0.08 and 0.1 here, while in the paper we examine 0.06-0.3)
    for c0 in range(8,10,2):
        df = pd.DataFrame(columns = ['data_size', 'price', 'base_precision', 'federal_precision', 'merge_precision'])
    
        res = {}
    
        # The value of cof_l should be specified by the user in advance
        cof_l = 200
        cof_c0 = c0/100
    
        alice_sampler = torch.utils.data.SubsetRandomSampler(alice_indices, torch.Generator())
        alice_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, sampler = alice_sampler, shuffle = False, **kwargs)
        base_model = copy.deepcopy(initial_model)
        get_model_param(args, base_model, device, alice_loader, valid_loader)
        base_precision, loss_0= test(args, base_model, device, test_loader)
        
        print('---------base_precision:'+str(base_precision)+'--------------')
        
        # Calculate the expected precision when the data price and the data size are both 0
        temp_r = 0
        temp_x = 0
        cof_p = (base_precision + 1)/math.exp(temp_r ** 2 /( 2 * cof_l * cof_c0)) 
    
        # Round1. The first round of data bidding is specified by the user
        temp_r = 1 
        temp_x = int((temp_r * 100) /( 2 * cof_c0))
        temp_p = cof_p * math.exp(temp_r ** 2 /( 2 * cof_l * cof_c0))-1
        
        rounds = 1
        while(True):
            print(rounds)
            score = {}
            score['data_size'] = temp_x
            score['price'] = temp_r
            bob_indices = indices[ : temp_x]     #
            total_indices = alice_indices + bob_indices
            alice_sampler = torch.utils.data.SubsetRandomSampler(alice_indices, torch.Generator())
            bob_sampler = torch.utils.data.SubsetRandomSampler(bob_indices, torch.Generator())
            total_sampler = torch.utils.data.SubsetRandomSampler(total_indices, torch.Generator())
            alice_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, sampler = alice_sampler, shuffle = False, **kwargs)
            bob_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, sampler = bob_sampler, shuffle = False, **kwargs)
            total_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, sampler = total_sampler, shuffle = False, **kwargs)
    
            model = copy.deepcopy(initial_model)
            base_model = copy.deepcopy(model)
            merge_model = copy.deepcopy(model)
    
            get_model_param(args, base_model, device, alice_loader, valid_loader)
            score['base_precision'], loss_1= test(args, base_model, device, test_loader)
    
            get_model_param(args, merge_model, device, total_loader, valid_loader)
            score['merge_precision'], loss_3 = test(args, merge_model, device, test_loader)
    
            model_val_loss = -1
            temp_model = np.nan
            while True:    
                alices_model = copy.deepcopy(model)
                bobs_model = copy.deepcopy(model)
    
                alice_best_params = get_model_param(args, alices_model, device, alice_loader,valid_loader)
                bob_best_parms = get_model_param(args, bobs_model , device, bob_loader,valid_loader)
    
                list1 = []
                for param in alice_best_params:
                    list1.append(param)
                list2 = []
                for param in bob_best_parms:
                    list2.append(param)
                new_parm = []
    
                for i, j in zip(list1,list2):
                    new_parm.append(len(alice_indices)*i/(len(alice_indices)+len(bob_indices))+len(bob_indices)*j/(len(alice_indices)+len(bob_indices)))
    
                k=0
                for parm in model.parameters():
                    parm.data = new_parm[k]
                    k=k+1
    
                acc_temp, loss_temp = test(args, model, device, valid_loader)
                if model_val_loss == -1:
                    model_val_loss = loss_temp
                    temp_model = copy.deepcopy(model)
                elif model_val_loss < loss_temp:
                    model = copy.deepcopy(temp_model)
                    break
                model_val_loss = loss_temp
                temp_model = copy.deepcopy(model)
    
            score['federal_precision'], loss_2 = test(args, model, device, test_loader)
            
            # Simple judgment to determine whether the result can be found in this round
            if rounds!=1:
                pass
            elif rounds == 1 and math.sqrt(cof_l * cof_c0 * math.log((1 + score['federal_precision'])/(1 + base_precision))) < 1 and  math.sqrt(2 * cof_l * cof_c0 * math.log( 2 / (1 + base_precision))) > 1:
                pass
            else:
                print(str(c0) + '---------no results-------'+'   '+ str(score['federal_precision']))
                break
                
            #  Judge whether the federated learning results meets the expectation(we set a range of the expected accuracy here)
            if score['federal_precision']> base_precision and score['federal_precision']<= temp_p + 0.001 and score['federal_precision']>= temp_p - 0.001:
                res['price'] =  temp_r
                res['data_size'] = temp_x
                res['income1'] = temp_r * temp_x /100 - cof_c0 * ((temp_x/100)** 2)
                res['income2'] = cof_l * math.log(1 + score['federal_precision']) - temp_r * temp_x /100
                df.loc[len(df)] = [score['data_size'], score['price'], score['base_precision'], score['federal_precision'], score['merge_precision']]
                results.loc[len(results)] = [res['data_size'], res['price'], res['income1'], res['income2']]
                break
            # Update the next round's parameter from the previous round's parameter
            cof_p = (score['federal_precision'] + 1)/math.exp(temp_r ** 2 /( 2 * cof_l * cof_c0)) 
            temp_r = math.sqrt((2 * cof_l * cof_c0) * math.log((temp_p + 1)/cof_p))
            temp_x = int((temp_r * 100)/(2 * cof_c0))
            temp_p = cof_p * math.exp(temp_r ** 2 /( 2 * cof_l * cof_c0))-1
            df.loc[len(df)] = [score['data_size'], score['price'], score['base_precision'], score['federal_precision'], score['merge_precision']]
            rounds = rounds + 1
            
        df.to_excel('results_500_200_{}.xlsx'.format(cof_c0),encoding='utf-8-sig',index=False)
        results.to_excel('results_500_200_0.08-0.1.xlsx', encoding='utf-8-sig',index=False)
        results.to_excel('results_500_200_0.08-0.1.xlsx', encoding='utf-8-sig',index=False)
    
    
