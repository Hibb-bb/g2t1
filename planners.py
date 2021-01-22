import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.ops import edge_softmax

class Planner(nn.Module):
    def __init__(self, glove_emb, glove_dim, inp_dim, out_dim):
        super(Planner, self).__init__()
        self.glove_dim = glove_dim
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.title_emb = glove_emb
        self.rnn = nn.GRU( inp_dim + glove_dim, out_dim , num_layers = 2, dropout = 0.3)

        self.title_enc = nn.GRU(glove_dim, glove_dim, num_layers = 2, dropout = 0.3, bidirectional = True)


    def selection(self, src_id, graph, random_walk = False):
        edges = graph.out_edges(src_id, form = 'eid')
        sg = dgl.edge_subgraph(graph, edges, preserve_nodes = True)
        sg.apply_edges(fn.u_mul_v('p', 'p', 'prob'))
        sg.edata['route'] = edge_softmax(sg, sg.edata['prob'])
        
        if(not random_walk):
            sg = dgl.sampling.select_topk(sg, 1, 'route')
        else:
            sg = dgl.sampling.random_walk(sg, length = 1, prob = 'route')
        
        return sg, sg.edges()


    def forward(self, graph,title_src, src_node_id,gru_state=None):
        # GRU Phase
        title_src = title_src.unsqueeze(1)
        title = self.title_emb(title_src) # 1*emb_dim
        out,hidden  = self.title_enc(title)
        title = torch.mean( hidden, dim = 0  )
        title = title.squeeze(0) # 1*emb
        cat_feat = torch.stack( [ torch.cat(( title, ai ), dim = -1) for ai in graph.ndata['h']])
        cat_feat = cat_feat.unsqueeze(1)
        if(gru_state == None):
            f, hx = self.rnn( cat_feat  )
        else:
            f, hx = self.rnn(cat_feat, gru_state)
        out = f
        graph.ndata['p'] = out # p = planning state
        pred = torch.zeros( graph.number_of_edges())
        # Walk Phase
        sg, new_edge = self.selection(src_node_id, graph)
        for i in sg.all_edges('all')[-1]:
            pred[i] = sg.edata['route'][i]

        new_edge_id = graph.edge_id(new_edge[0], new_edge[1])
        return graph, new_edge_id, new_edge[1], hx, pred# graph, edge index, new start node, gru state, prediction, decode_subgraph


