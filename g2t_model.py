import torch
import torch.nn
from random import random
import dgl



class G2TModel(nn.Module):
    def __init__(self,encoder, planner, decoder, plan_tf_ratio = 1, dec_tf_ratio):
        super(G2TModel ,self).__init__()

        self.encoder = encoder
        self.planner = planner
        self.decoder = decoder
        self.plan_optim = torch.optim.Adam([
            {'params':self.encoder.parameters()},
            {'params':self.planner.parameters()}
            ])

    def decode(self, graph, tgt, hx= None):
        out_tgt = []
        for i in range(tgt.size(0)):
            pred, hx = self.decoder(graph, hx):
            out_tgt.append(pred)
        return [torch.stack(out_tgt), 'decode']
        

    def plan(self, graph, title_tgt, src_node, tgt_edge_idx, hx=None):
        graph, next_edge_id, next_node, hx, pred = self.planner(graph, title_tgt, src_node, hx)
        return [graph, next_edge_id, next_node, hx, pred, 'plan']

    def forward(self, graph, src_node = None, edge_order=None, title_tgt=None, tgt=None,hx = None,  mode = 'plan'):

        if(mode == 'encode'):
            return self.encoder(graph)

        elif(mode == 'plan'):
            return self.plan(graph, title_tgt, src_node, hx)

        elif(mode == 'decode'):
            return self.decode(graph, tgt, hx)

        else:
            return False

        graph = self.encoder(graph)
        edge_pred = torch.zeros(len(edge_order))
        hx = None
        src_node = edge_order[0][0]
        for i in range(edge_order):
            graph, next_edge_id, next_node, hx, node_feat = self.planner(graph, title_tgt, src_node, hx)
            edge_pred[next_edge_id] = 1
            if(random() < plan_tf_ratio):
                src_node = edge_order[i][0]
            else:
                src_node = next_node
        
        out_tgt = []        
        for i in range(tgt.size(0)):
            pred, hx = self.decoder(graph, hx)
            out_tgt.append(pred)

        return torch.stack(out_tgt), edge_pred
