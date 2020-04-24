import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, remove_self_loops, add_self_loops, softmax,scatter_
#from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot, zeros
import sys
import inspect
is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec
special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

class GMNlayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GMNlayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.out_channels = out_channels
        self.fmessage = nn.Linear(3*in_channels, out_channels)
        self.fnode = torch.nn.GRUCell(2*out_channels, out_channels, bias=True)
        self.__match_args__ = getargspec(self.match)[0][1:]
        self.__special_match_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__match_args__)
                                 if arg in special_args]
        self.__match_args__ = [
            arg for arg in self.__match_args__ if arg not in special_args
        ]

    '''def propagate(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            #print(arg)
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]
        out = self.message(*message_args)
        out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        #print(out.size())
        out = self.update(out, *update_args)
        return out'''
    def propagate_match(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        match_args = []
        #print(self.__special_match_args__)
        #print(self.__match_args__)
        #print(ij.keys())
        for arg in self.__match_args__:
            #print(arg)
            #print(arg[-2:])
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    match_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    match_args.append(tmp)
                #print(tmp)
            else:
                match_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_match_args__:
            if arg[-2:] in ij.keys():
                match_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                match_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]
        #print(match_args)
        out_attn = self.match(*match_args)
        #print(out_attn.size())
        out_attn = scatter_(self.aggr, out_attn, edge_index[i], dim_size=size[i])
        #print(out_attn.size())
        out_attn = self.update(out_attn, *update_args)
        #out=torch.cat([out,out_attn],dim=1)
        #print(out.size())
        return out_attn

    def forward(self, x, edge_index,edge_index_attn):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)

        # Step 3-5: Start propagating messages.
        m=self.propagate(edge_index,size=(x.size(0), x.size(0)), x=x,edge_weight=None)
        u=self.propagate_match(edge_index_attn,size=(x.size(0), x.size(0)),x=x)
        m=torch.cat([m,u],dim=1)
        h=self.fnode(m,x)
        return h

    def message(self, x_i, x_j, edge_index,size,edge_weight=None):
        # x_j has shape [E, out_channels]
        # Step 3: Normalize node features.
        #print(x_i.size(),x_j.size())
        if edge_weight==None:
            edge_weight=torch.ones(x_i.size(0),x_i.size(1)).cuda()
            m=F.relu(self.fmessage(torch.cat([x_i,x_j,edge_weight],dim=1)))
        else:
            m=F.relu(self.fmessage(torch.cat([x_i,x_j],dim=1)))
        return m

    def match(self, edge_index_i, x_i, x_j, size_i):
        #x_j = x_j.view(-1, 1, self.out_channels)
        #alpha = torch.dot(x_i, x_j)
        #print(edge_index_i.size())
        #print(x_i.size(),x_j.size())
        alpha=torch.sum(x_i*x_j, dim=1)
        #alpha=torch.bmm(x_i.unsqueeze(1), x_j.unsqueeze(2))
        #print(alpha.size())
        size_i=x_i.size(0)
        alpha = softmax(alpha, edge_index_i, size_i)
        #print(alpha.size())
        '''c = torch.ones(A, B) * 2
        v = torch.randn(A, B, C)
        print(c)
        print(v)
        print(c[:,:, None].size())
        d = c[:,:, None] * v'''
        return alpha[:,None]*x_j
        #return x_j* alpha.view(-1, 1, 1)
        #return (x_i-x_j)* alpha.view(-1, 1, 1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out

class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def propagate(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        print(self.__message_args__)
        for arg in self.__message_args__:
            print(arg)
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    print(tmp.size())
                    message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        out = self.message(*message_args)
        out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        out = self.update(out, *update_args)
        return out
    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))
        #print(x.size())
        #print(size)
        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        print(edge_index_i.size())#size[E,]
        # Compute attention coefficients.
        #print(x_i.size(),x_j.size())
        #print(size_i)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
class GMNnet(torch.nn.Module):
    def __init__(self,vocablen,embedding_dim,num_layers):
        super(GMNnet, self).__init__()
        self.num_layers=num_layers
        self.embed=nn.Embedding(vocablen,embedding_dim)
        #self.gmn=nn.ModuleList([GMNlayer(embedding_dim,embedding_dim) for i in range(num_layers)])
        self.gmnlayer=GMNlayer(embedding_dim,embedding_dim)

    def forward(self, data):
        x, edge_index, edge_index2, batch = data.x, data.edge_index, data.edge_index2, data.batch
        x = self.embed(x)
        x = x.squeeze(1)
        for i in range(self.num_layers):
            x=self.gmnlayer(x,edge_index, edge_index2)
        #for layer in self.gmn:
            #x=layer(x,edge_index, edge_index2)