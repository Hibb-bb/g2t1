import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import dgl
from dgl.nn import GATConv

class LocalAttn(nn.Module):
    
    def __init__(self, gate_nn, feat_nn=None, inp_dim = None):
        super(LocalAttn, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
        self.inp_dim = inp_dim

    def forward(self, graph, feat):
        
        with graph.local_scope():
            feat = feat.view(-1, self.inp_dim)
            gate = self.gate_nn(feat)
            # assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = dgl.softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            out = graph.ndata['r']
            graph.ndata.pop('r')
            
            return out




class LocalAggr(nn.Module):
    def __init__(self,inp_dim, hid_dim):
        super(LocalAggr, self).__init__()
        inp_dim = hid_dim 
        self.wq = nn.Linear(inp_dim, hid_dim)
        self.wk = nn.Linear(inp_dim, hid_dim)
        self.prop = LocalAttn(self.wq, self.wk, inp_dim)

    def forward(self,g):
        feat = g.ndata['h']
        feat = feat.unsqueeze(-1)
        out = self.prop(g, feat)
        otu = out.squeeze(-1)
        return out 

class GlobalAggr(nn.Module):
    def __init__(self,inp_dim, hid_dim):
        super(GlobalAggr, self).__init__()
        # self.gat = GATConv(inp_dim, hid_dim, num_heads = 8, allow_zero_in_degree = True)
        self.wk = nn.Parameter(torch.Tensor(inp_dim, hid_dim))
        nn.init.xavier_uniform_( self.wk , gain = nn.init.calculate_gain('relu'))

        self.wq = nn.Parameter(torch.Tensor(inp_dim, hid_dim))
        nn.init.xavier_uniform_( self.wq , gain = nn.init.calculate_gain('relu'))
        
        self.wg = nn.Parameter(torch.Tensor(inp_dim, hid_dim))
        nn.init.xavier_uniform_( self.wg , gain = nn.init.calculate_gain('relu'))

        self.softmax = nn.Softmax( dim = 0 )

    def forward(self,g):
        # out = torch.sum(self.gat(g, g.ndata['h']), dim = 1)
        h = g.ndata['h']
        h = h.squeeze(1)
        query = torch.mm(h, self.wq) # emb * n
        key = torch.mm(h, self.wk)
        attn = torch.mm( key, torch.transpose(query, 0, 1))
        attn = self.softmax(attn)
        out = torch.mm( attn, torch.mm(h, self.wg)) # (n, n) * (n, inp) * (inp, hid) 
        return out.unsqueeze(1)



class CGE(nn.Module):
    def __init__(self,emb_layer, inp_dim, hid_dim, layer_num = 1):
        super(CGE, self).__init__()
        self.emb_layer = emb_layer
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.layer_num = layer_num
        # self.ln = nn.LayerNorm(self.hid_dim)
        self._build_layer()
        self.act = nn.LeakyReLU()
        self.ffn = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU()
                )
    
    def _build_layer(self):
        self.le = nn.ModuleList([ LocalAggr(self.hid_dim, self.hid_dim) for _ in range(self.layer_num )])
        ge_lst = []
        ge_lst.append(GlobalAggr(self.inp_dim, self.hid_dim))
        for i in range(self.layer_num - 1):
            ge_lst.append(GlobalAggr(self.hid_dim, self.hid_dim))
        self.ge = nn.ModuleList(ge_lst)

    def forward(self,g):
        
        nodes = g.ndata['x']
        edges = g.edata['x']
        nodes = self.emb_layer(nodes)
        edges = self.emb_layer(edges)
        g.ndata['h'] = nodes # n*emb_dim
        g.edata['h'] = edges # e*emb_dim

        g.update_all( fn.u_mul_e('h', 'h', 'h'), fn.sum('h', 'h'))
        g.ndata['h'] = g.ndata['h'] + nodes # not sure to add self feat or not
        g.ndata['h'] = self.act(g.ndata['h'])
        for i in range(self.layer_num):
            g.ndata['h'] = self.ge[i](g) # n * hid
            g.ndata['h'] = self.ffn(g.ndata['h'])
            g.ndata['h'] = self.le[i](g)
            g.ndata['h'] = self.act(g.ndata['h'])
        print(g.ndata['h'].shape)
        g.ndata['enc'] = g.ndata['h']
        return g




class Encoder(nn.Module):
    def __init__(self, emb_layer, emb_dim, out_dim):
        super(Encoder, self).__init__()
        self.emb_layer = emb_layer

        self.w1 = nn.Parameter( torch.Tensor(emb_dim, out_dim )  ) # node w
        self.bias1 = nn.Parameter( torch.Tensor(out_dim))
        
        nn.init.xavier_uniform_(self.w1, gain = nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.bias1, gain = nn.init.calculate_gain('sigmoid'))
        
        
        self.w2 = nn.Parameter( torch.Tensor(emb_dim, out_dim )  ) # edge w
        self.bias2 = nn.Parameter( torch.Tensor(out_dim))
        
        nn.init.xavier_uniform_(self.w2, gain = nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.bias2, gain = nn.init.calculate_gain('sigmoid'))
        

        self.wq = nn.Parameter( torch.Tensor(out_dim,out_dim ))
        self.wk = nn.Parameter( torch.Tensor(out_dim,out_dim ))
        self.relu = nn.ReLU()

    def forward(self,g):
        # local aggr      
        nodes = g.ndata['x']
        edges = g.edata['x']
        edge_idx = g.edges() # 2 * edges
        edge_id = g.edges(form ='all' )[2] # edge num * 1
        nodes = self.emb_layer(nodes) # node_num * emb_dim
        edges = self.emb_layer(edges) # edge_num * emb_dim

        nodes = self.w1*nodes + self.bias1
        nodes = self.relu(nodes)

        edges = self.w2*edges + self.bias2
        edges = self.relu(edges)

        # g.ndata['h'] = dgl.ops.u_mul_e_sum(g, nodes, edges)
        g.ndata['h'] = nodes
        g.edata['h'] = edges
        
        # global aggr
        

        g.update_all( fn.u_mul_e('h', 'h', 'h'), fn.sum('h', 'h'))

        return g


        
        

        
    

