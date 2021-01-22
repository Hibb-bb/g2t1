import torch
import torch.nn as nn
from tqdm import tqdm
import json
import json_lines
import numpy as np

def str2emb(data):
	
	vec = torch.tensor([float(val) for val in data], requires_grad=True)

	return vec



def load_entity_embedding(dp):
    emb_dict = {}
    missing_rel = [ 'R:P1963', 'P529', 'R:P1687', 'P625' ,'P1963', 'R:P529', 'P1687', 'R:P625' ]
    with open(dp, 'r') as f:
        for l in f:
            l = l.split()
            ent_id = l[0]
            ent_emb = l[1:]
            ent_emb = str2emb(ent_emb)
            emb_dict[ent_id] = ent_emb

    return emb_dict

def load_relation_embedding(dp):
    emb_dict = {}
    '''
    R:P1963
    key error
    P529
    key error
    R:P1687
    key error
    P625
    '''
    # missing_rel = [ 'R:P1963', 'P529', 'R:P1687', 'P625' ,'P1963', 'R:P529', 'P1687', 'R:P625' ]
    with open(dp, 'r') as f:
        for l in f:
            l = l.split()
            rel_id = l[0]
            rel_emb = l[1:]
            rel_emb = str2emb(rel_emb)
            emb_dict[rel_id] = rel_emb
    return emb_dict

# ent_emb = load_entity_embedding('../data/linked_wikitext//embeddings.entities.txt')
# rel_emb = load_relation_embedding('../data/linked_wikitext//embeddings.relations.txt')
# print(len(rel_emb))
def build_universal_dict(datapath = './data/linked_wikitext/linkedwiki'):
    ent2num = {}
    ent_num = 0
    ent2num["@@NEW@@"] = 0
    ent_num+=1
    with open(datapath + '/train-ordered-tgt.jsonl') as f:
        for l in json_lines.reader(f):
            t = l['tgt']
            all_ids = t['ids']
            for ids in all_ids:
                for i, ents in enumerate(ids):
                    for j, ent in enumerate(ents):
                        if( ent not in list(ent2num.keys()) ):
                            ent2num[ent] = ent_num
                            ent_num+=1
    f.close()
    with open(datapath + '//dev-ordered-tgt.jsonl') as f:
        for l in json_lines.reader(f):
            t = l['tgt']
            all_ids = t['ids']
            for ids in all_ids:
                for i, ents in enumerate(ids):
                    for j, ent in enumerate(ents):
                        if( ent not in list(ent2num.keys()) ):
                            ent2num[ent] = ent_num
                            ent_num+=1
    f.close()
    with open(datapath + '//test-ordered-tgt.jsonl') as f:
        for l in json_lines.reader(f):
            t = l['tgt']
            all_ids = t['ids']
            for ids in all_ids:
                for i, ents in enumerate(ids):
                    for j, ent in enumerate(ents):
                        if( ent not in list(ent2num.keys()) ):
                            ent2num[ent] = ent_num
                            ent_num+=1
    f.close()
    return ent2num



def get_triplets(datapath):
    triplets = []
    ent_num = 0
    with open(datapath) as f:
        for l in json_lines.reader(f):
            line = []
            t = l['tgt']
            all_ids = t['ids']
            for ids in all_ids:
                for i in ids:
                    line.append(i)
            triplets.append(line)
    return triplets
    f.close()

# t = get_triplets()

# ent2num = build_universal_dict()
# print(ent2num['Q2349106'])
# print(len(ent2num))
# print(ent2num)
'''
datapath  = '../data/linked_wikitext/linkedwiki/train-ordered-tgt.jsonl'
R:P1963
key error
P529
key error
R:P1687
key error
P625
'''
# print(rel_emb['R:P1963'])
# print(ent2num['R:P1963'])

