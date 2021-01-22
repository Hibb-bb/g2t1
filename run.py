import torch
import torch.nn
from data_loader import *
from model.embeddings import *
from model.encoders import *
from model.model import G2TModel
from model.planners import *
import argparse




def get_args():
    parser = argparse.ArgumentParser(description='Graph To Text')
    """
    Data Processing
    """
    parser.add_argument('--dataset-name', type=str, default='LinkedWikiText2',
                    help='name of the dataset')


    """
    Model Parameters
    """
    parser.add_argument('--glove-dim', type = int, default = 200,
                    help = 'the dimension of glove embedding, for title embedding and decoder input')
    parser.add_argument('--embedding-dim', type=int, default='256',
                        help='embedding dimension and encoder input dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='dimension of encoder output')
    parser.add_argument('--planning-dim', type=int, default= 512,
                        help='planning dim')


    """
    Training Parameters
    """
    parser.add_argument('--epoch', type = int, default = 5,
                    help = 'training epoch')
    



class Trainer:
    def __init__(self, args):

        self.model = G2TModel(CGE(LinkedWikiEmbedding(7108), 256, 512),Planner(GloveEmbedding(200), 200, 512, 512))
        self.train_loader = BatchedGraphs('./data/linked_wikitext/linkedwiki/train-ordered-tgt.jsonl', './data/linked_wikitext/train.jsonl')
        self.valid_loader = BatchedGraphs('./data/linked_wikitext/linkedwiki/dev-ordered-tgt.jsonl', './data/linked_wikitext/valid.jsonl')
        self.test_loader = BatchedGraphs('./data/linked_wikitext/linkedwiki/test-ordered-tgt.jsonl', './data/linked_wikitext/test.jsonl')
        self.planner_opt = torch.optim.Adam(  [
        {'params':self.model.encoder.parameters()},
        {'params':self.model.planner.parameters()}
        ])
        self.model_opt = torch.optim.Adam(model.parameters(), lr =0.005 )
        self.dec_cri = nn.NLLLoss()
        self.plan_cri = nn.CrossEntropyLoss()
        self.epoch = args.epoch
        self.plan_tf_ratio = 1
        self.decode_tf_ratio = 0.5
        
    def planning(self, graph, title_tgt):
        total_loss = 0
        hx = None
        for i in range(len(edge_order)):
            src_node = edge_order[i][0]
            out = self.model(graph, src_node = src_node,  hx = hx, title_tgt= title_tgt, mode = 'plan')
            loss = self.plan_cri(out[-2], i )
            total_loss+=loss
            loss.backward()
            self.planner_opt.step()
            

        return out, total_loss/(len(edge_order))

    def decoding(self, subgraph, tgt, hx=None):
        
        out = self.model( subgraph, tgt = tgt, hx = hx )
        loss = self.dec_cri(out[0], tgt)
        loss.backward()
        self.model_opt.step()
        return loss

    def train_epoch(self):
        total_plan_loss = 0
        total_dec_loss = 0
        for data in self.train_loader:
            graph = data[0]
            sent_len = data[1]
            edge_order = data[2]
            title_tgt = data[3]
            tgt = data[4]
            cur_edge_id = 0
            for sent_idx, sent in enumerate(sent_len):
                plan_out, plan_loss = self.planning(graph,title_tgt)
                graph = plan_out[0]
                sg_id = [i for i in range(cur_edge_id, cur_edge_id+sent)]
                dec_sg =  graph.edge_subgraph(graph, sg_id)
                loss = self.decoding(dec_sg, tgt[sent_idx])
                total_dec_loss+=loss
                total_plan_loss+=plan_loss
                loss.backward()
                self.model_opt.step()
                cur_edge_id = sent



                      
#                



a = BatchedGraphs('./data/linked_wikitext/linkedwiki/dev-ordered-tgt.jsonl', 
        './data/linked_wikitext/valid.jsonl', 
        './data/linked_wikitext/linkedwiki/dev-surfaces.txt')

print('graph',a[0][0])
print('sent len',a[0][1])
print('graph orders',a[0][2])
print('title',a[0][3])
print('align',a[0][4])

'''
emb_layer = LinkedWikiEmbedding(7108)
# enc = Encoder(emb_layer,256, 256)


cge = CGE(emb_layer, 256, 512)

# data : 0 : graph
#        1 : sent len
#        2 : token order
#        3 : title

glove = GloveEmbedding(50)
p = Planner(glove, 50, 512, 512)


for data in a:
    g = data[0]
    # print(g.ndata['x'])
    g = cge(g)
    e, ns, gru, node  = p(g, data[3], 0)
    print(g.ndata['h'])
    print(node)
    break
# print(enc)


# glove = GloveEmbedding(50)
# p = Planner(glove, 50, 512,512)
'''
