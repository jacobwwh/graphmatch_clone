import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import sys
import argparse
from tqdm import tqdm, trange
import pycparser
from createclone import createast,creategmndata
import models
from torch_geometric.data import Data, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--data_setting", default='11')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--lr", default=0.001)
args = parser.parse_args()
 
device=torch.device('cuda:0')
astdict,vocablen,vocabdict=createast()
traindata,validdata,testdata=creategmndata('11small',astdict,vocablen,vocabdict,device)
#print(traindata)
#trainloder=DataLoader(traindata,batch_size=1)

model=models.GMNnet(vocablen,embedding_dim=100,num_layers=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion=nn.CosineEmbeddingLoss()
def create_batches(data):
    #random.shuffle(data)
    batches = [data[graph:graph+args.batch_size] for graph in range(0, len(data), args.batch_size)]
    return batches
epochs = trange(args.num_epochs, leave=True, desc = "Epoch")
for epoch in epochs:# without batching
    print(epoch)
    batches=create_batches(traindata)
    totalloss=0.0
    main_index=0.0
    for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
        optimizer.zero_grad()
        batchloss= 0
        for data,label in batch:
            prediction=model(data)
            batchloss=batchloss+criterion(prediction[0],prediction[1],label)
        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        totalloss+=loss
        main_index = main_index + len(batch)
        loss=totalloss/main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss,5))

    #for start in range(0, len(traindata), args.batch_size):
        #batch = traindata[start:start+args.batch_size]
        #epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
'''for batch in trainloder:
    batch=batch.to(device)
    print(batch)
    quit()
    time_start=time.time()
    model.forward(batch)
    time_end=time.time()
    print(time_end-time_start)
    quit()'''