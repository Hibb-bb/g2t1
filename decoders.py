import torch.nn as nn
import torch
import dgl
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim, method = 'dot'):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim*2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, dec_out, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(dec_out, enc_outs)
        return F.softmax(attn_energies, dim=0)

    def dot(self, dec_out, enc_outs):
        return torch.sum(dec_out*enc_outs, dim=2)

    def general(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out*energy, dim=2)

    def concat(self, dec_out, enc_outs):
        dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((dec_out, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)


class DecRNN(nn.Module):
    def __init__(self, vocab_size, inp_dim, emb_dim,emb_hid_dim, hid_dim, n_layers, use_birnn, 
                 dropout= 0.2, tied = True, pretrained_emb = None ):
        super(DecRNN, self).__init__()
        
        
        if(not pretrained_emb):
            self.emb = nn.Embedding(vocab_size, emb_dim)
        else:
            self.emb = pretrained_emb
        
        self.emb_rnn = nn.GRUCell( emb_dim, emb_dim )
        self.mlp1 = nn.Sequential(
                nn.Linear(emb_dim, emb_hid_dim),
                nn.Sigmoid()
                )

        self.rnn = nn.LSTM(emb_hid_dim + inp_dim , hid_dim , n_layers,bidirectional = use_birnn  )

        self.w = nn.Linear(hid_dim*2, hid_dim)
        self.attn = Attention(hid_dim, 'dot')

        self.out_projection = nn.Linear(hid_dim, vocab_size)

        if tied: 
            self.out_projection.weight = self.emb.weight

        self.dropout = nn.Dropout(dout)

    def forward(self, g,  hidden):
        # inputs = inputs.unsqueeze(0)
        tgt = g.ndata['name']
        enc_outs = g.ndata['enc']
        tgt_emb = self.dropout( self.mlp1(self.emb(tgt)))
        tgt_emb = torch.mean(tgt_emb, dim = 0).squeeze(1)
        # embs = self.dropout(self.emb(tgt_name))
        node_feat = g.ndata['h']
        node_feat = torch.cat([ node_feat, tgt_emb ],dim = 1 ) # concat node feature and token name embedding

        dec_out, hidden = self.rnn(node_feat, hidden)

        attn_weights = self.attn(dec_out, enc_outs).transpose(1, 0)
        enc_outs = enc_outs.transpose(1, 0)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outs)
        cats = self.w(torch.cat((dec_out, context.transpose(1, 0)), dim=2))
        pred = self.out_projection(cats.tanh().squeeze(0))
        return pred, hidden



d = DecRNN(100, 10, 20, 20, 40, 1, True)


'''

class Seq2seqAttn(nn.Module):
    def __init__(self, args, fields, device):
        super().__init__()
        self.src_field, self.tgt_field = fields
        self.src_vsz = len(self.src_field[1].vocab.itos)
        self.tgt_vsz = len(self.tgt_field[1].vocab.itos)
        self.decoder = DecRNN(self.tgt_vsz, args.embed_dim, args.hidden_dim, 
                              args.n_layers, args.bidirectional, args.dropout,
                              args.attn, args.tied)
        self.device = device
        self.n_layers = args.n_layers
        self.hidden_dim = args.hidden_dim
        self.use_birnn = args.bidirectional

    def forward(self, srcs,plan_out, tgts=None, maxlen=100, tf_ratio=0.0):
        slen, bsz = srcs.size()
        tlen = tgts.size(0) if isinstance(tgts, torch.Tensor) else maxlen
        tf_ratio = tf_ratio if isinstance(tgts, torch.Tensor) else 0.0
       
        enc_outs, hidden = self.encoder(srcs)

        dec_inputs = torch.ones_like(srcs[0]) * 2 # <eos> is mapped to id=2
        outs = []

        if self.use_birnn:
            def trans_hidden(hs):
                hs = hs.view(self.n_layers, 2, bsz, self.hidden_dim)
                hs = torch.stack([torch.cat((h[0], h[1]), 1) for h in hs])
                return hs
            hidden = tuple(trans_hidden(hs) for hs in hidden)

        for i in range(tlen):
            preds, hidden = self.decoder(dec_inputs, hidden, enc_outs)
            outs.append(preds)
            use_tf = random.random() < tf_ratio
            dec_inputs = tgts[i] if use_tf else preds.max(1)[1]
        return torch.stack(outs)
'''
