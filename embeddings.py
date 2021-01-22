import torch
import torch.nn as nn
from model.utils import build_universal_dict, load_entity_embedding, load_relation_embedding
import numpy as np




class GloveEmbedding(nn.Module):
    def __init__(self, dim):
        super(GloveEmbedding, self).__init__()
        self.dim = dim
        if(self.dim == 50):
            dp = './model/glove/glove.6B.50d.txt'
        if(self.dim == 100):
            dp = './model/glove/glove.6B.100d.txt'
        if(self.dim == 200):
            dp = './model/glove/glove.6B.200d.txt'
        if(self.dim == 300):
            dp = './model/glove/glove.6B.300d.txt'

        emb_mat = self._load_emb(dp)
        self.emb = nn.Embedding.from_pretrained(emb_mat)


    def _load_emb(self, dp):
        word2id = {}
        id2word = {}
        word2id['<UNK>'] = 0
        word2id['<EOS>'] = 2
        word2id['<PAD>'] = 1
        id2word[0] = '<UNK>'
        id2word[2] = '<EOS>'
        id2word[1] = '<PAD>'
        word_num = 3
        emb_mat = torch.zeros( 400003, self.dim )
        emb_mat[2] = torch.rand(self.dim)
        with open(dp, 'r') as f:
            lines = f.readlines()
        f.close()

        for line in lines:
            l = line.split()
            l[-1] = l[-1].replace('\n', '')
            word2id[l[0]] = word_num
            id2word[word_num] = l[0]
            vect = np.array(l[1:]).astype(np.float)
            emb_mat[word_num] = torch.from_numpy(vect)
            word_num +=1

        return emb_mat

    def forward(self,x):
        return self.emb(x)

glove = GloveEmbedding(50)

class NormalEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim = 256):
        super(NormalEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        return self.emb(x)


class LinkedWikiEmbedding(nn.Module):
    def __init__(self,vocab_size, emb_dim = 256):
        super(LinkedWikiEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        emb_mat = self.load_emb(0)
        self.emb = nn.Embedding.from_pretrained(emb_mat)

    def load_emb(self,datapath = None):
        print('loading linkedwiki pretrained embeddings')
        self.ent2num = build_universal_dict()
        weights_matrix = torch.zeros((self.vocab_size, self.emb_dim))
        self.ent2emb = load_entity_embedding('./data/linked_wikitext/embeddings.entities.txt')
        print('ent emb len',len(self.ent2emb)) # 
        self.rel2emb = load_relation_embedding('./data/linked_wikitext/embeddings.relations.txt')
        print('rel emb len',len(self.rel2emb)) # 7506
        for ent in list(self.ent2num.keys()):
            if(ent in self.ent2emb):
                weights_matrix[ self.ent2num[ent] ] = self.ent2emb[ent]
            elif(ent in self.rel2emb):
                weights_matrix[ self.ent2num[ent] ] = self.rel2emb[ent]
            else:
                # print("cannot find",ent,"embedding, replace it with random")
                if( ent == "@@NEW@@"):
                    print("relation @@NEW@@ has embedding with one")
                    weights_matrix[ self.ent2num[ent] ] = torch.ones(self.emb_dim, requires_grad=True)
                else:
                    print("cannot find",ent,"embedding, replace it with random")
                    weights_matrix[ self.ent2num[ent] ] = torch.rand(self.emb_dim, requires_grad=True)
        print(weights_matrix)
        return weights_matrix

    def forward(self,x):
        return self.emb(x)


# e = LinkedWikiEmbedding(7108)
# i = torch.zeros(4, dtype = torch.long)
# i[1] = 20
# i[2] = 7000
# print(e(i).shape)
