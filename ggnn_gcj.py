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
from createclone_java import createast,creategmndata,createseparategraph
import models
from torch_geometric.data import Data, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default='gcj')
parser.add_argument("--graphmode", default='astandnext')
parser.add_argument("--nextsib", default=False)
parser.add_argument("--ifedge", default=False)
parser.add_argument("--whileedge", default=False)
parser.add_argument("--foredge", default=False)
parser.add_argument("--blockedge", default=False)
parser.add_argument("--nexttoken", default=False)
parser.add_argument("--nextuse", default=False)
parser.add_argument("--data_setting", default='0')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()
 
device=torch.device('cuda:0')
#device=torch.device('cpu')
astdict,vocablen,vocabdict=createast()
treedict=createseparategraph(astdict, vocablen, vocabdict,device,mode=args.graphmode,nextsib=args.nextsib,ifedge=args.ifedge,whileedge=args.whileedge,foredge=args.foredge,blockedge=args.blockedge,nexttoken=args.nexttoken,nextuse=args.nextuse)
traindata,validdata,testdata=creategmndata(args.data_setting,treedict,vocablen,vocabdict,device)
print(len(traindata))
#trainloder=DataLoader(traindata,batch_size=1)
num_layers=int(args.num_layers)
model=models.GGNN(vocablen,embedding_dim=100,num_layers=num_layers,device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion=nn.CosineEmbeddingLoss()
criterion2=nn.MSELoss()
def create_batches(data):
    #random.shuffle(data)
    batches = [data[graph:graph+args.batch_size] for graph in range(0, len(data), args.batch_size)]
    return batches

def test(dataset):
    #model.eval()
    count=0
    correct=0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results=[]
    for data,label in dataset:
        label=torch.tensor(label, dtype=torch.float, device=device)
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=data
        x1=torch.tensor(x1, dtype=torch.long, device=device)
        x2=torch.tensor(x2, dtype=torch.long, device=device)
        edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
        if edge_attr1!=None:
            edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
            edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)
        data1=[x1,edge_index1,edge_attr1]
        data2=[x2,edge_index2,edge_attr2]
        prediction1=model(data1)
        prediction2=model(data2)
        output=F.cosine_similarity(prediction1,prediction2)
        results.append(output.item())
        prediction = torch.sign(output).item()

        if prediction>args.threshold and label.item()==1:
            tp+=1
            #print('tp')
        if prediction<=args.threshold and label.item()==-1:
            tn+=1
            #print('tn')
        if prediction>args.threshold and label.item()==-1:
            fp+=1
            #print('fp')
        if prediction<=args.threshold and label.item()==1:
            fn+=1
            #print('fn')
    print(tp,tn,fp,fn)
    p=0.0
    r=0.0
    f1=0.0
    if tp+fp==0:
        print('precision is none')
        return
    p=tp/(tp+fp)
    if tp+fn==0:
        print('recall is none')
        return
    r=tp/(tp+fn)
    f1=2*p*r/(p+r)
    print('precision')
    print(p)
    print('recall')
    print(r)
    print('F1')
    print(f1)
    return results

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
            label=torch.tensor(label, dtype=torch.float, device=device)
            #print(len(data))
            #for i in range(len(data)):
                #print(i)
                #data[i]=torch.tensor(data[i], dtype=torch.long, device=device)
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=data
            x1=torch.tensor(x1, dtype=torch.long, device=device)
            x2=torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1!=None:
                edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)
            data1=[x1,edge_index1,edge_attr1]
            data2=[x2,edge_index2,edge_attr2]
            prediction1=model(data1)
            prediction2=model(data2)
            batchloss=batchloss+criterion(prediction1,prediction2,label)
            #cossim=F.cosine_similarity(prediction1,prediction2)
            #batchloss=batchloss+criterion2(cossim,label)
        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        totalloss+=loss
        main_index = main_index + len(batch)
        loss=totalloss/main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
    #test(validdata)
    devresults=test(validdata)
    devfile=open('ggnngcjresult/'+args.graphmode+'_dev_epoch_'+str(epoch+1),mode='w')
    for res in devresults:
        devfile.write(str(res)+'\n')
    devfile.close()
    #test(testdata)
    testresults=test(testdata)
    resfile=open('ggnngcjresult/'+args.graphmode+'_epoch_'+str(epoch+1),mode='w')
    for res in testresults:
        resfile.write(str(res)+'\n')
    resfile.close()

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
