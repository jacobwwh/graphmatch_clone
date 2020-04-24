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
from createclone_bcb import createast, creategmndata, createseparategraph
import models
from torch_geometric.data import Data, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default='gcj')
parser.add_argument("--graphmode", default='astandnext')
parser.add_argument("--nextsib", default=True)
parser.add_argument("--ifedge", default=False)
parser.add_argument("--whileedge", default=False)
parser.add_argument("--foredge", default=False)
parser.add_argument("--blockedge", default=False)
parser.add_argument("--nexttoken", default=False)
parser.add_argument("--nextuse", default=False)
parser.add_argument("--data_setting", default='11')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=5)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()

# device=torch.device('cuda:0')
device = torch.device('cpu')
astdict, vocablen, vocabdict = createast()
treedict = createseparategraph(astdict, vocablen, vocabdict, device, mode=args.graphmode, nextsib=args.nextsib,
                               ifedge=args.ifedge, whileedge=args.whileedge, foredge=args.foredge,
                               blockedge=args.blockedge, nexttoken=args.nexttoken, nextuse=args.nextuse)
traindata, validdata, testdata = creategmndata(args.data_setting, treedict, vocablen, vocabdict, device)
print("numTrainData:")
print(len(traindata))
# trainloder=DataLoader(traindata,batch_size=1)
num_layers = int(args.num_layers)
model = models.GMNnet(vocablen, embedding_dim=100, num_layers=num_layers, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CosineEmbeddingLoss()
criterion2 = nn.MSELoss()


def create_batches(data):
    # random.shuffle(data)
    batches = [data[graph:graph + args.batch_size] for graph in range(0, len(data), args.batch_size)]
    return batches


def test(dataset,epoch):
    # model.eval()
    ff=open("recordFATTOld"+str(epoch)+".txt",'w')
    count = 0
    correct = 0
    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0
    tp = []
    tn = []
    fp = []
    fn = []
    for yi in range(0,18):
        tp.append(0)
        tn.append(0)
        fp.append(0)
        fn.append(0)
    results = []
    for data, label in dataset:
        label = torch.tensor(label, dtype=torch.float, device=device)
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
        x1 = torch.tensor(x1, dtype=torch.long, device=device)
        x2 = torch.tensor(x2, dtype=torch.long, device=device)
        edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
        if edge_attr1 != None:
            edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
            edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
        data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
        prediction = model(data)
        output = F.cosine_similarity(prediction[0], prediction[1])
        results.append(output.item())
        prediction = torch.sign(output).item()
        ff.write(str(output.item()))
        ff.write(" ")

    ff.close()
    return results


epochs = trange(args.num_epochs, leave=True, desc="Epoch")
for epoch in epochs:  # without batching
    print(epoch)
    batches = create_batches(traindata)
    totalloss = 0.0
    main_index = 0.0
    zz=0
    for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
        optimizer.zero_grad()
        batchloss = 0
        # zzz=0
        # zz+=1
        # if zz>10:
        #     break
        for data, label in batch:
            # zzz+=1
            # print(zzz)
            label = torch.tensor(label, dtype=torch.float, device=device)
            # print(len(data))
            # for i in range(len(data)):
            # print(i)
            # data[i]=torch.tensor(data[i], dtype=torch.long, device=device)
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
            x1 = torch.tensor(x1, dtype=torch.long, device=device)
            x2 = torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1 != None:
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
            data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
            prediction = model(data)
            # batchloss=batchloss+criterion(prediction[0],prediction[1],label)
            cossim = F.cosine_similarity(prediction[0], prediction[1])
            batchloss = batchloss + criterion2(cossim, label)
        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        totalloss += loss
        main_index = main_index + len(batch)
        loss = totalloss / main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
    # test(validdata)
    #devresults = test(validdata)
    # devfile = open('gmnbcbresultnew/' + args.graphmode + '_dev_epoch_' + str(epoch + 1), mode='w')
    # for res in devresults:
    #     devfile.write(str(res) + '\n')
    # devfile.close()
    print("start test:")
    testresults = test(testdata,epoch)

    # resfile = open('gmnbcbresultnew/' + args.graphmode + '_epoch_' + str(epoch + 1), mode='w')
    # for res in testresults:
    #     resfile.write(str(res) + '\n')
    # resfile.close()
    # torch.save(model,'gmnmodels/gmnbcb'+str(epoch+1))
    # for start in range(0, len(traindata), args.batch_size):
    # batch = traindata[start:start+args.batch_size]
    # epochs.set_description("Epoch (Loss=%g)" % round(loss,5))

'''for batch in trainloder:
    batch=batch.to(device)
    print(batch)
    quit()
    time_start=time.time()
    model.forward(batch)
    time_end=time.time()
    print(time_end-time_start)
    quit()'''
