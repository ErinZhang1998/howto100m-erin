from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
from args import get_args
from model import Net
from metrics import compute_metrics, print_computed_metrics
from gensim.models.keyedvectors import KeyedVectors
import pickle
import glob
from lsmdc_dataloader import LSMDC_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader
from youcook_dataloader import Youcook_DataLoader
from epic_dataloader import Epic_DataLoader
import os
import numpy as np

args = get_args()
if args.verbose:
    print(args)

assert args.pretrain_path != '', 'Need to specify pretrain_path argument'

print('Loading word vectors: {}'.format(args.word2vec_path))
we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
print('done')


# generate start_idx for all segments
if args.eval_epic:
    root_path = '/raid/xiaoyuz1/EPIC'
    gt_path = os.path.join(root_path, 'howto100m_groundTruth/verb')
    args.features_path_2D = os.path.join(root_path, 'Features/2D')
    args.features_path_3D = os.path.join(root_path, 'Features/3D')
    start_idx = dict()
    for vid in os.listdir(gt_path):
        gt = open(os.path.join(gt_path, vid), 'r').read().split('\n')[:-2]
        tmp = list(np.arange(len(gt)))
        tmp = np.array(list(filter(lambda x: (x==0 or (gt[x] != gt[x-1])), tmp)))
        start_idx[vid.strip('.txt')] = tmp

    dataset = Epic_DataLoader(
      features_path = args.features_path_2D,
      features_path_3D = args.features_path_3D,
      start_idx = start_idx,      # {video_id : array of start idx of a segmentation}
      gt_path = gt_path,
      we=we,
      we_dim=args.we_dim,
      max_words=args.max_words,
      train=False
    )
    dataloader_epic = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )

if args.eval_youcook:
    dataset_val = Youcook_DataLoader(
        data=args.youcook_val_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.eval_lsmdc:
    dataset_lsmdc = LSMDC_DataLoader(
        csv_path=args.lsmdc_test_csv_path,
        features_path=args.lsmdc_test_features_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
    dataloader_lsmdc = DataLoader(
        dataset_lsmdc,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.eval_msrvtt:
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.msrvtt_test_csv_path,
        features_path=args.msrvtt_test_features_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=3000,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
net = Net(
    video_dim=args.feature_dim,
    embd_dim=args.embd_dim,
    we_dim=args.we_dim,
    max_words=args.max_words,
)
net.eval()
net.cuda()

if args.verbose:
    print('Starting evaluation loop ...')

def Eval_retrieval(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating Text-Video retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            text = data['text'].cuda()
            video = data['video'].cuda()
            print(video.shape)
            print(text.shape)
            vid = data['video_id']
            m = model(video, text)
            m  = m.cpu().detach().numpy()
            print(m.shape)
            print(m)
            metrics = compute_metrics(m)
            print_computed_metrics(metrics)

# def Eval_epic(model, eval_dataloader, dataset_name):
#     model.eval()
#     print('Evaluating Text-Video retrieval on {} data'.format(dataset_name))
#     with th.no_grad():
#         for i_batch, data in enumerate(eval_dataloader):
#             text = data['text'].cuda()
#             video = data['video'].cuda()
#             print(video.shape)
#             print(text.shape)
#             vid = data['video_id']
#             m = model(video, text)
#             m  = m.cpu().detach().numpy()
#             print(m.shape)
#             print(m)
#             metrics = compute_metrics(m)
#             print_computed_metrics(metrics)

args.pretrain_path = os.path.join('/raid/xiaoyuz1/EPIC', 'howto100m', 'model/fine_tune_result_epoch_50.pth')
all_checkpoints = glob.glob(args.pretrain_path)

for c in all_checkpoints:
    print('Eval checkpoint: {}'.format(c))
    print('Loading checkpoint: {}'.format(c))
    net.load_checkpoint(c)
    if args.eval_epic:
      Eval_retrieval(net, dataloader_epic, 'EpicKitchens')
    if args.eval_youcook:
        Eval_retrieval(net, dataloader_val, 'YouCook2')
    if args.eval_msrvtt:
        Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')
    if args.eval_lsmdc:
        Eval_retrieval(net, dataloader_lsmdc, 'LSMDC')
