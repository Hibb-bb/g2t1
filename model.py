import torch.nn as nn
import dgl
# from encoders import RGCNLayer


class G2TModel(nn.Module):
    def __init__(self, encoder, planner, glove_emb):
        super(G2TModel, self).__init__()
        self.encoder = encoder
        self.glove_emb = glove_emb
        self.planner = planner
        # self.decoder = 
    def forward(self,graph):
        graph = self.encoder(graph)

        return graph


