import numpy as np
import torch
from torchvision import transforms as T
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34 ,resnet50
import pandas as pd
from sklearn.utils import shuffle
import copy
import math

from PIL import Image
from collections import OrderedDict
import cv2
from sklearn.decomposition import PCA
import time

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import os
import random
import itertools
import json

hyperparams = [[18,34,50],[0],[32,64],[1,2,3]] 
              #[[Resnet Depth],[Unsupervised Epochs],[Batch Size],[Random_state]]
l_hyperparams = list(itertools.product(*hyperparams))
              #[(18, 0, 32, 1),(18, 0, 32, 2),(18, 0, 32, 3),...,(50, 0, 64, 1),(50, 0, 64, 2),(50, 0, 64, 3)]
l_l_hyperparams = [list(x) for x in l_hyperparams] 
              #[[18, 0, 32, 1],[18, 0, 32, 2],[18, 0, 32, 3],...,[50, 0, 64, 1],[50, 0, 64, 2],[50, 0, 64, 3]]


for rd, ne, b, rs in l_l_hyperparams:
  resnet_depth = rd
  num_epochs = ne
  batch = b
  r_state = rs

  train_prop=0.9
  linear_prop= 2/9
  num_epochs_linear = 10
  val_prop = 0.05
  test_prop = 0.05

  #Resnet Training
  TRAINING = True
  if num_epochs == 0:
    TRAINING = False
  #Linear Training
  LINEAR = True
  if num_epochs_linear == 0:
    LINEAR = False


  path_cnv_1 = ''
  path_pcv_1 = ''
  path_cnv_2 = ''
  path_pcv_2 = ''
  path_cnv_k = ''
  path_pcv_3 = ''
  path_pcv_4 = ''

  model_name='MCv2_R'+str(resnet_depth)+'_'+str(num_epochs)+'_'+str(batch)+'_'+str(r_state)
  print(model_name)
  
  #Create folders
  if os.path.exists('') == False:
    os.mkdir('')
    os.mkdir('')
    os.mkdir('')
    os.mkdir('')

  #Create list of patients' ID
  names_cnv_1 = list(set([int(f.split('_')[0]) for f in os.listdir(path_cnv_1)]))
  names_cnv_1.sort()
  names_pcv_1 = list(set([int(f.split('_')[0]) for f in os.listdir(path_pcv_1)]))
  names_pcv_1.sort()
  names_cnv_2 = list(set([int(f.split('_')[0]) for f in os.listdir(path_cnv_2)]))
  names_cnv_2.sort()
  names_pcv_2 = list(set([int(f.split('_')[0]) for f in os.listdir(path_pcv_2)]))
  names_pcv_2.sort()
  names_cnv_k = list(set([int(f.split('-')[1]) for f in os.listdir(path_cnv_k)]))
  names_cnv_k.sort()

  #Create dict patients' ID as keys and their files as values
  def get_files(l_id,path,sep,ind):
    d = dict()
    l = [f for f in os.listdir(path)]
    for i in l_id:
      dl = []
      for j in l:
        if j.split(sep)[ind]==str(i):
          dl.append(j)
      dl.sort()
      d[i] = dl
    return d

  d_cnv_1 = get_files(names_cnv_1,path_cnv_1,'_',0)
  d_pcv_1 = get_files(names_pcv_1,path_pcv_1,'_',0)
  d_cnv_2 = get_files(names_cnv_2,path_cnv_2,'_',0)
  d_pcv_2 = get_files(names_pcv_2,path_pcv_2,'_',0)
  d_cnv_k = get_files(names_cnv_k,path_cnv_k,'-',1)

  #Split train linear val test
  random.seed(r_state)
  def train_lin_test(l_id,d,r_train,r_lin,r_val,p):
    train_len = round(len(l_id)*r_train)
    linear_len = round(train_len*r_lin)
    val_len = round(len(l_id)*r_val);
    test_len = len(l_id)-train_len-val_len
    train_id = random.sample(l_id,train_len)
    train_id.sort()
    linear_id = random.sample(train_id,linear_len)
    linear_id.sort()
    val_id = random.sample(list(set(l_id)-set(train_id)),val_len)
    val_id.sort()
    test_id = random.sample(list(set(l_id)-set(train_id)-set(val_id)),test_len)
    test_id.sort()
    train = []  ;linear = [] ;test = []  ;val = [] 

    for i in train_id:
      if i in d.keys():
        train+=[p+x for x in d[i]]

    for i in linear_id:
      if i in d.keys():
        linear+=[p+x for x in d[i]]

    for i in val_id:
      if i in d.keys():
        val+=[p+x for x in d[i]]    

    for i in test_id:
      if i in d.keys():
        test+=[p+x for x in d[i]]

    return train,linear,val,test,train_id,linear_id,val_id,test_id

  train_cnv_1,linear_cnv_1,val_cnv_1,test_cnv_1,train_cnv_1_id,linear_cnv_1_id,val_cnv_1_id,test_cnv_1_id = train_lin_test(names_cnv_1,d_cnv_1,train_prop,linear_prop,val_prop,path_cnv_1)
  train_pcv_1,linear_pcv_1,val_pcv_1,test_pcv_1,train_pcv_1_id,linear_pcv_1_id,val_pcv_1_id,test_pcv_1_id = train_lin_test(names_pcv_1,d_pcv_1,train_prop,linear_prop,val_prop,path_pcv_1)
  train_cnv_2,linear_cnv_2,val_cnv_2,test_cnv_2,train_cnv_2_id,linear_cnv_2_id,val_cnv_2_id,test_cnv_2_id  = train_lin_test(names_cnv_2,d_cnv_2,train_prop,linear_prop,val_prop,path_cnv_2)
  train_pcv_2,linear_pcv_2,val_pcv_2,test_pcv_2,train_pcv_2_id,linear_pcv_2_id,val_pcv_2_id,test_pcv_2_id = train_lin_test(names_pcv_2,d_pcv_2,train_prop,linear_prop,val_prop,path_pcv_2)
  train_cnv_k,linear_cnv_k,val_cnv_k,test_cnv_k,train_cnv_k_id,linear_cnv_k_id,val_cnv_k_id,test_cnv_k_id = train_lin_test(names_cnv_k,d_cnv_k,train_prop,linear_prop,val_prop,path_cnv_k)
  train_pcv_3 = [path_pcv_3+f for f in os.listdir(path_pcv_3)]
  linear_pcv_4 = [path_pcv_4+f for f in os.listdir(path_pcv_4)]

  rand_state = r_state
  train_files = shuffle(train_cnv_1+train_pcv_1+train_cnv_2+train_cnv_k+train_pcv_2+train_pcv_3,random_state=rand_state)
  linear_files = shuffle(linear_cnv_1+linear_pcv_1+linear_cnv_k+linear_cnv_2+linear_pcv_2+linear_pcv_4,random_state=rand_state)
  val_files = shuffle(val_cnv_1+val_pcv_1+val_cnv_2+val_pcv_2,random_state=rand_state)
  test_files = shuffle(test_cnv_1+test_pcv_1+test_cnv_2+test_pcv_2,random_state=rand_state)

  train_labels = [0 if x.split('/')[6] == 'CNV' else 1 for x in train_files]
  linear_labels = [0 if x.split('/')[6] == 'CNV' else 1 for x in linear_files]
  val_labels = [0 if x.split('/')[6] == 'CNV' else 1 for x in val_files]
  test_labels = [0 if x.split('/')[6] == 'CNV' else 1 for x in test_files]
  print('Train: '+str(len(train_labels)))
  print('Linear: '+str(len(linear_labels)))
  print('Val: '+str(len(val_labels)))
  print('Test: '+str(len(test_labels)))
  tc=0
  tp=0
  for i in [train_labels,linear_labels,val_labels,test_labels]:
    n_c = 0
    n_p = 0
    for j in i:
      if j == 0:
        n_c += 1
      else:
        n_p += 1
    tc+=n_c
    tp+=n_p
    print(n_c)
    print(n_p)
  print('tc= '+str(tc))
  print('tp='+str(tp))
  print(len(val_cnv_1_id+val_cnv_2_id+val_cnv_k_id),len(test_cnv_1_id+test_cnv_2_id))
  print(len(val_pcv_1_id+val_pcv_2_id+val_cnv_k_id),len(test_pcv_1_id+test_pcv_2_id))
  #Create device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  def get_color_distortion(s=1.0):
      color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
      rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
      
      # p is the probability of grayscale, here 0.2
      rnd_gray = T.RandomGrayscale(p=0.2)
      color_distort = T.Compose([rnd_color_jitter, rnd_gray])
      
      return color_distort

  # this is the dataset class

  class MyDataset(Dataset):
      def __init__(self, filenames, labels, mutation=False):
          #self.root_dir = [root_dir+'/'+x.split('/')[1]+'/'+x.split('/')[2] for x in filenames]
          self.file_names = filenames
          self.labels = labels
          self.mutation = mutation

      def __len__(self):
          return len(self.file_names)

      def tensorify(self, img):
          res = T.ToTensor()(img)
          res = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(res)
          return res

      def mutate_image(self, img):
          res = T.RandomResizedCrop(224)(img) #224
          res = get_color_distortion(1)(res)
          return res

      def __getitem__(self, idx):
          if torch.is_tensor(idx):
              idx = idx.tolist()

          #img_name = os.path.join(self.root_dir, self.file_names[idx])
          img_name = self.file_names[idx]
          image = Image.open(img_name).convert('RGB')
          label = self.labels[idx]
          image = T.Resize((508, 490))(image) #250,250

          if self.mutation:
              image1 = self.mutate_image(image)
              image1 = self.tensorify(image1)
              image2 = self.mutate_image(image)
              image2 = self.tensorify(image2)
              sample = {'image1': image1, 'image2': image2, 'label': label}
          else:
              image = T.Resize((508, 490))(image)
              image = self.tensorify(image)
              sample = {'image': image, 'label': label}

          return sample
  training_dataset_mutated = MyDataset( train_files, train_labels, mutation=True)
  linear_dataset = MyDataset( linear_files, linear_labels, mutation=False)
  val_dataset = MyDataset(val_files, val_labels, mutation=False)
  testing_dataset = MyDataset( test_files, test_labels, mutation=False)
  dataloader_training_dataset_mutated = DataLoader(training_dataset_mutated, batch_size=batch, shuffle=True, num_workers=2)
  dataloader_training_dataset = DataLoader(linear_dataset, batch_size=64, shuffle=False, num_workers=2)
  dataloader_val_dataset = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
  dataloader_testing_dataset = DataLoader(testing_dataset, batch_size=64, shuffle=False, num_workers=2)
  K = 8192
  ptr = True
  # defining our deep learning architecture
  if resnet_depth == 18:
    resnetq = resnet18(pretrained=ptr) #True
  if resnet_depth == 34:
    resnetq = resnet34(pretrained=ptr) #True
  if resnet_depth == 50:
    resnetq = resnet50(pretrained=ptr) #True

  #Projection head
  classifier = nn.Sequential(OrderedDict([
      ('fc1', nn.Linear(resnetq.fc.in_features, 100)),
      ('added_relu1', nn.ReLU(inplace=True)),
      ('fc2', nn.Linear(100, 50)),
      ('added_relu2', nn.ReLU(inplace=True)),
      ('fc3', nn.Linear(50, 25))
  ]))

  resnetq.fc = classifier
  resnetk = copy.deepcopy(resnetq)

  # moving the resnet architecture to device
  resnetq.to(device)
  resnetk.to(device)
 
  # Code for NT-Xent Loss function, explained in more detail in the article

  τ = 0.05

  def loss_function(q, k, queue):

      N = q.shape[0]
      C = q.shape[1]

      pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),τ))
      neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),τ)), dim=1)
      denominator = neg + pos

      return torch.mean(-torch.log(torch.div(pos,denominator)))
  # Defining data structures for storing training info

  losses_train = []

  flag = 0

  queue = None

  # using SGD optimizer
  optimizer = optim.SGD(resnetq.parameters(), lr=0.01, momentum=0.9)
  #load pretrained model, optimizer and training losses file if model.pth file is available
  load_old_model = False
  if load_old_model:
    if (os.path.isfile("")):
      resnetq.load_state_dict(torch.load("{}{}".format(model_name)))
      resnetk.load_state_dict(torch.load("{}{}".format(model_name)))
      optimizer.load_state_dict(torch.load("{}{}".format(model_name)))
      
      for param_group in optimizer.param_groups:
          param_group['weight_decay'] = 1e-6
          param_group['lr'] = 0.0003

      temp = np.load("{}{}".format(model_name))
      losses_train = list(temp['arr_0'])
      queue = torch.load("{}".format(model_name))
  if queue is None:
      while True:

          with torch.no_grad():
              for (_, sample_batched) in enumerate(dataloader_training_dataset_mutated):            

                  xk = sample_batched['image2']
                  xk = xk.to(device)
                  k = resnetk(xk)
                  k = k.detach()

                  k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

                  if queue is None:
                      queue = k
                  else:
                      if queue.shape[0] < K:
                          queue = torch.cat((queue, k), 0)    
                      else:
                          flag = 1
                  
                  if flag == 1:
                      break

          if flag == 1:
              break
  momentum = 0.999
  time1 = []
  current_ep=0    

  def get_mean_of_list(L):
    return sum(L) / len(L)

  # Boolean variable on whether to perform training or not 
  # Note that this training is unsupervised, it uses the NT-Xent Loss function

  if TRAINING == False:   

    # Store model and optimizer files
    torch.save(resnetq.state_dict(), '{0}{1}'.format(model_name,model_name))
    torch.save(resnetk.state_dict(), '{0}{1}'.format(model_name,model_name))
    torch.save(optimizer.state_dict(), '{0}{1}'.format(model_name,model_name))
    np.savez("{0}{1}".format(model_name,model_name), np.array(losses_train))
    torch.save(queue, '{0}{1}'.format(model_name,model_name))

  if TRAINING:
    start = time.time()
    # get resnet in train mode
    resnetq.train()

    # run a for loop for num_epochs
    for epoch in range(num_epochs):
      start = time.time()
      # a list to store losses for each epoch
      epoch_losses_train = []

      # run a for loop for each batch
      for (_, sample_batched) in enumerate(dataloader_training_dataset_mutated):
        # zero out grads
        optimizer.zero_grad()

        # retrieve xq and xk the two image batches
        xq = sample_batched['image1']
        xk = sample_batched['image2']

        # move them to the device
        xq = xq.to(device)
        xk = xk.to(device)

        # get their outputs
        q = resnetq(xq)
        k = resnetk(xk)
        k = k.detach()

        q = torch.div(q,torch.norm(q,dim=1).reshape(-1,1))
        k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

        # get loss value
        loss = loss_function(q, k, queue)
            
        # put that loss value in the epoch losses list
        epoch_losses_train.append(loss.cpu().data.item())

        # perform backprop on loss value to get gradient values
        loss.backward()

        # run the optimizer
        optimizer.step()

        # update the queue
        queue = torch.cat((queue, k), 0) 

        if queue.shape[0] > K:
            queue = queue[256:,:]

        # update resnetk
        for θ_k, θ_q in zip(resnetk.parameters(), resnetq.parameters()):
            θ_k.data.copy_(momentum*θ_k.data + θ_q.data*(1.0 - momentum))


      # append mean of epoch losses to losses_train, essentially this will reflect mean batch loss
      losses_train.append(get_mean_of_list(epoch_losses_train))

      # Store model and optimizer files
      torch.save(resnetq.state_dict(), '{0}{1}'.format(model_name,model_name))
      torch.save(resnetk.state_dict(), '{0}{1}'.format(model_name,model_name))
      torch.save(optimizer.state_dict(), '{0}{1}'.format(model_name,model_name))
      np.savez("{0}{1}".format(model_name,model_name), np.array(losses_train))
      torch.save(queue, '{0}{1}'.format(model_name,model_name))
      end = time.time()
      time1.append((end-start)/60)
      current_ep = current_ep+1
      print(current_ep,'/',num_epochs)
      print('{} mins'.format((end - start)/60))

  print('Total Training Time with Unlabelled Data: ',sum(time1))
  if TRAINING:
    fig = plt.figure(figsize=(10, 10))
    sns.set_style('darkgrid')
    plt.title('Training Loss', fontsize=18)
    plt.xlabel('Epochs')
    plt.ylabel('NT-Xent Loss')
    plt.plot(losses_train)
    plt.legend(['Training Losses'])
    plt.savefig('{0}{0}'.format(model_name))
    plt.show()

  # removing the projection head
  #if len(nn.Sequential(*list(resnetq.fc.children()))) == 5:
  resnetq.fc = nn.Sequential(*list(resnetq.fc.children())[:-1])


  # Boolean variable to control whether to train the linear classifier or not

  class LinearNet(nn.Module):

      def __init__(self):
          super(LinearNet, self).__init__()
          self.fc1 = torch.nn.Linear(50, 2)



      def forward(self, x):
          x = self.fc1(x)
          return(x)

  current_ep_lin = 0

  if LINEAR:
    # getting our linear classifier
    linear_classifier = LinearNet()

    # moving it to device
    linear_classifier.to(device)

    # using SGD as a linear optimizer
    linear_optimizer = optim.SGD(linear_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)

    #number of epochs
    

    # Boolean variable to control training of linear classifier
    LINEAR_TRAINING = True

    # Defining data structures to store train and test info for linear classifier
    losses_train_linear = []
    acc_train_linear = []
    losses_test_linear = []
    acc_test_linear = []

    # a variable to keep track of the maximum test accuracy, will be useful to store 
    # model parameters with the best test accuracy
    max_test_acc = 0


    # Run a for loop for training the linear classifier
    for epoch in range(num_epochs_linear):
      start = time.time()
      y_prob_accum = []
      y_probs = list()
      if LINEAR_TRAINING:
        # run linear classifier in train mode
        linear_classifier.train()

        # a list to store losses for each batch in an epoch
        epoch_losses_train_linear = []
        epoch_acc_train_num_linear = 0.0
        epoch_acc_train_den_linear = 0.0

        # for loop for running through each batch
        for (_, sample_batched) in enumerate(dataloader_training_dataset):
          # get x and y from the batch
          x = sample_batched['image']
          y_actual = sample_batched['label']
          #y_actual = torch.tensor([Classes_Map[i] for i in y_actual])
          # move them to the device
          x = x.to(device)
          y_actual  = y_actual.to(device)

          with torch.no_grad():
            # get output from resnet architecture
            y_intermediate = resnetq(x)

          # zero the grad values
          linear_optimizer.zero_grad()

          # run y_intermediate through the linear classifier
          y_predicted = linear_classifier(y_intermediate)
          
          # get the cross entropy loss value
          loss = nn.CrossEntropyLoss()(y_predicted, y_actual)

          # add the obtained loss value to this list
          epoch_losses_train_linear.append(loss.data.item())
          
          # perform backprop through the loss value
          loss.backward()

          # call the linear_optimizer step function
          linear_optimizer.step()

          # get predictions and actual values to cpu  
          pred = np.argmax(y_predicted.cpu().data, axis=1)
          actual = y_actual.cpu().data

          #update the numerators and denominators of accuracy
          epoch_acc_train_num_linear += (actual == pred).sum().item()
          epoch_acc_train_den_linear += len(actual)

          x = None
          y_intermediate = None
          y_predicted = None
          sample_batched = None

        # update losses and acc lists    
        losses_train_linear.append(get_mean_of_list(epoch_losses_train_linear))
        acc_train_linear.append(epoch_acc_train_num_linear / epoch_acc_train_den_linear)
      train_acc = epoch_acc_train_num_linear / epoch_acc_train_den_linear    
      current_ep_lin = current_ep_lin+1
        
      print(str(current_ep_lin)+'/'+str(num_epochs_linear))
      print('Train acc: '+str(train_acc))
      # run linear classifier in eval mode
      linear_classifier.eval()

      # essential variables to keep track of losses and acc
      epoch_losses_test_linear = []
      epoch_acc_test_num_linear = 0.0
      epoch_acc_test_den_linear = 0.0
      Y_TRUE = []
      Y_PRED = []
      # run a for loop through each batch
      for (_, sample_batched) in enumerate(dataloader_testing_dataset):

        x = sample_batched['image']
        y_actual = sample_batched['label']
        #y_actual = torch.tensor([Classes_Map[i] for i in y_actual])

        x = x.to(device)
        y_actual  = y_actual.to(device)

        with torch.no_grad():
            y_intermediate = resnetq(x)

        y_predicted = linear_classifier(y_intermediate)
        #print(y_predicted)
        m = nn.Softmax(dim=1)

        
        y_prob = m(y_predicted)
        y_prob_accum += y_prob.squeeze().tolist()
        y_probs += list(np.argmax(y_prob.cpu().data,axis =1 ))

        loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
        epoch_losses_test_linear.append(loss.data.item())
        
        pred = np.argmax(y_predicted.cpu().data, axis=1)
        #pred = np.argmax(y_prob, axis=1)

        actual = y_actual.cpu().data
        epoch_acc_test_num_linear += (actual == pred).sum().item()
        epoch_acc_test_den_linear += len(actual)
        Y_TRUE.append(actual)
        Y_PRED.append(pred)
      # calculate test_acc
      test_acc = epoch_acc_test_num_linear / epoch_acc_test_den_linear
      losses_test_linear.append(get_mean_of_list(epoch_losses_test_linear))
      acc_test_linear.append(epoch_acc_test_num_linear / epoch_acc_test_den_linear)
      print('Test acc: '+str(test_acc))


      if test_acc >= max_test_acc:
        # save the model only when test_acc exceeds the current max_test_acc

        max_test_acc = test_acc
        Y_TRUE_1=[]
        Y_PRED_1=[]
        Y_TRUE_1.append(Y_TRUE)
        Y_PRED_1.append(Y_PRED)
        y_probs_max = y_probs
        y_prob_accum_max = y_prob_accum
        torch.save(linear_classifier.state_dict(), '{0}{0}'.format(model_name))
        torch.save(linear_optimizer.state_dict(), '{0}{0}'.format(model_name))


          # save data structures
        np.savez("{0}{0}".format(model_name), np.array(losses_train_linear))
        np.savez("{0}{0}".format(model_name), np.array(losses_test_linear))
        np.savez("{0}{0}".format(model_name), np.array(acc_train_linear))
        np.savez("{0}{0}".format(model_name), np.array(acc_test_linear))


      end = time.time()
      print('Time {} mins'.format((end - start)/60))
  end_t = time.time()
  print(' ')
  print('Total time {} mins'.format((end_t - start)/60))
  yp = np.concatenate( Y_PRED, axis=0 )
  yp
  yt = np.concatenate(Y_TRUE, axis=0 )
  yt
  yt_max = np.concatenate( Y_TRUE_1[0], axis=0 )
  yt_max
  yp_max = np.concatenate( Y_PRED_1[0], axis=0 )
  yp_max
  from sklearn.metrics import classification_report

  target_names = ['CNV', 'PCV']

  print(classification_report(yt_max, yp_max,target_names=target_names))
  max_test_acc
  if LINEAR:
    X = np.arange(1,num_epochs_linear+1)

    plt.figure(figsize=(10, 10))
    sns.set_style('darkgrid')
    plt.plot(losses_train_linear)
    plt.plot(losses_test_linear)
    plt.title('Loss')
    plt.legend(['Training Losses', 'Validating Losses'])
    plt.savefig('{0}{0}'.format(model_name))
    plt.show()

    plt.figure(figsize=(10, 10))
    sns.set_style('darkgrid')
    plt.plot(acc_train_linear)
    plt.plot(acc_test_linear)
    plt.title('Accuracy')
    plt.legend(['Training Accuracy', 'Validating Accuracy'])
    plt.savefig('{0}{0}'.format(model_name))
    plt.show()

    print("Epoch completed")

  # roc curve and auc
  from sklearn.datasets import make_classification
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import roc_curve
  from sklearn.metrics import roc_auc_score
  from matplotlib import pyplot
  from sklearn.metrics import f1_score
  def to_labels(pos_probs, threshold):
	  return (pos_probs >= threshold).astype('int')
  # generate a no skill prediction (majority class)
  ns_probs = [0 for _ in range(len(yt))]
  l_probs = np.array(y_prob_accum_max)[:,1]
  #lr_probs = lr_probs[:, 1]
  # calculate scores
  ns_auc = roc_auc_score(yt, ns_probs)
  l_auc = roc_auc_score(yt, l_probs)
  # summarize scores
  print('No Skill: ROC AUC=%.3f' % (ns_auc))
  print('Mocov2: ROC AUC=%.3f' % (l_auc))
  # calculate roc curves
  ns_fpr, ns_tpr, _ = roc_curve(yt, ns_probs)
  l_fpr, l_tpr, thresholds = roc_curve(yt, l_probs)
  # calculate the g-mean for each threshold
  gmeans = np.sqrt(l_tpr * (1-l_fpr))
  # locate the index of the largest g-mean
  ix = np.argmax(gmeans)
  print('Optimal Threshold for ROC Curve')
  print('Best Threshold=%.9f, G-Mean=%.9f, F-Score=%.5f' % (thresholds[ix], gmeans[ix], f1_score(yt_max, to_labels(l_probs,thresholds[ix]))))
  pyplot.figure(figsize=(10, 10))
  # plot the roc curve for the model
  pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
  pyplot.plot(l_fpr, l_tpr, marker='.', label='Mocov2')
  pyplot.scatter(l_fpr[ix], l_tpr[ix], marker='o', color='black', label='Best')
  # axis labels
  pyplot.xlabel('False Positive Rate')
  pyplot.ylabel('True Positive Rate')
  # show the legend
  pyplot.legend()

  pyplot.title(model_name)
  # show the plot


  plt.savefig('{0}'.format(model_name))
  pyplot.show()
  def perf_measure(y_actual, y_hat):
      TP = 0
      FP = 0
      TN = 0
      FN = 0

      for i in range(len(y_hat)): 
          if y_actual[i]==y_hat[i]==1:
            TP += 1
          if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
          if y_actual[i]==y_hat[i]==0:
            TN += 1
          if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

      return(TP, FP, TN, FN)

  TP,FP,TN,FN = perf_measure(yt_max, to_labels(l_probs,thresholds[ix]))
  TP,FP,TN,FN
  ACC = (TP+TN)/ (TP+TN+FP+FN) 
  Sensitivity = TP / (TP + FN)
  Specifity = TN/(TN+FP)
  PPV = TP/(TP+FP)
  #NPV = TN/(FN+TN)
  if FN+TN == 0:
    NPV = 'inf'
  else:
    NPV = TN/(FN+TN)
    
  AUC = l_auc
  F1 = f1_score(yt_max, to_labels(l_probs,thresholds[ix]))

  true_pred_results = {'Name':model_name,'True':[str(x) for x in list(yt_max)],'Predicted':[str(x) for x in list(to_labels(l_probs,thresholds[ix]))],'TP':TP,'FP':FP,
                      'TN':TN,'FN':FN,'Sensitivity':Sensitivity,'Specifity':Specifity,'PPV':PPV,'NPV':NPV,'ACC':ACC,'AUC':l_auc,'F1 max acc':F1,
                      'num_epochs_linear' : num_epochs_linear,'unsup_training_time (mins)':sum(time1),'num_epochs_linear' : num_epochs_linear,
                      'linear_training_time(mins)':(end_t - start)/60,'train_prop':train_prop,'linear_prop': linear_prop,'Val_cnv_patient_n':len(val_cnv_1_id+val_cnv_2_id+val_cnv_k_id),'Test_cnv_patient_n':len(test_cnv_1_id+test_cnv_2_id+test_cnv_k_id),
                        'Val_pcv_patient_n':len(val_pcv_1_id+val_pcv_2_id),'Test_pcv_patient_n':len(test_pcv_1_id+test_pcv_2_id),'Prob':y_prob_accum_max,'Threshold':thresholds[ix],'G-Means':gmeans[ix]}

  with open("{0}".format(model_name), 'w') as fp:
          json.dump(true_pred_results, fp)