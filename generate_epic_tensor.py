import pickle
import os
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import re
import torch
from tqdm import tqdm
from model import Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chunk_idx', type=int)
parser.add_argument('--chunk_size', type=int, default=10000)
parser.add_argument('--word2vec_path', type=str, required=True)
parser.add_argument('--pkl_train', type=str, required=True)
parser.add_argument('--pkl_val', type=str, required=True)
parser.add_argument('--howto100m_pretrained_path', type=str, required=True)
parser.add_argument('--output_file_name', type=str, required=True)
args = parser.parse_args()

we_dim = 300
max_words = 20
def zero_pad_tensor(tensor, size):
    if len(tensor) >= size:
        return tensor[:size]
    else:
        zero = np.zeros((size - len(tensor), we_dim), dtype=np.float32)
        return np.concatenate((tensor, zero), axis=0)

word2vec_path = args.word2vec_path
print('Loading word vectors: {}'.format(word2vec_path))
we = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
print('done')


# verb_path = '/home/xiaoyuz1/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
# pd.read_csv(verb_path, sep=',')
# verb_list = []
# verb_list_d_rev = dict()
# csv_np = pd.read_csv(verb_path, sep=',')['instances'].to_numpy()
# class_np = pd.read_csv(verb_path, sep=',')['id'].to_numpy()
# for L,class_id in zip(csv_np, class_np):
#     l = []
#     for s in L.split(","):
#         s_cleaned = s.translate(str.maketrans("'][","   ")).strip()
#         verb_list_d_rev[s_cleaned] = class_id
#         verb_list.append(s_cleaned)


pkl_train = pd.read_pickle(args.pkl_train)
pkl_val = pd.read_pickle(args.pkl_val)
all_narration = np.concatenate([pkl_train['narration'].to_numpy(), pkl_val['narration'].to_numpy()])
all_verb_class = np.concatenate([pkl_train['verb_class'].to_numpy(), pkl_val['verb_class'].to_numpy()])

words_feats = []
for s in all_narration:
    s = str(s)
    words = re.findall(r"[\w']+", s)
    words = [word for word in words if word in we.vocab] #index_to_key
    words_feat = None
    if words:
        words_feat =  torch.from_numpy(zero_pad_tensor(we[words], max_words))
    else:
        words_feat =  torch.zeros(max_words, we_dim) 
    words_feats.append(words_feat.unsqueeze(dim=0))

net = Net(
    video_dim=4096,
    embd_dim=6144,
    we_dim=300,
    n_pair=1,
    max_words=20,
    sentence_dim=-1,
)
net.cuda()
net.eval()
print(net)
net.load_checkpoint(args.howto100m_pretrained_path)

text_embeddings = []
total = len(words_feats)
chunk_size = args.chunk_size
num_chunks = total // chunk_size + 1
chunk_idx = args.chunk_idx
for idx in tqdm(range(chunk_size)):
    if chunk_size*chunk_idx + idx >= total:
        continue
    # if idx % 10000 ==0:
    #     print(chunk_size*chunk_idx + idx)
    words_feat = words_feats[chunk_size*chunk_idx + idx]
    words_feat = words_feat.cuda()
    # print(words_feat.shape)
    words_feat = net.text_pooling(words_feat)
    # print(words_feat.shape)
    text = net.GU_text(words_feat)
    text = text.detach().cpu()
    text_embeddings.append(text)
    torch.cuda.empty_cache()

text_embeddings = torch.cat(text_embeddings, dim=0)
all_verb_class = all_verb_class[chunk_size*chunk_idx: chunk_size*chunk_idx + chunk_size]
with open(args.output_file_name, 'wb+') as f:
    pickle.dump([text_embeddings,all_verb_class], f)