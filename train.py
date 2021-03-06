from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from args import get_args
import random
import os
from youtube_dataloader import Youtube_DataLoader
from youcook_dataloader import Youcook_DataLoader
from model import Net
from metrics import compute_metrics, print_computed_metrics, compute_epic_metrics
from loss import MaxMarginRankingLoss, TripletLoss
from gensim.models.keyedvectors import KeyedVectors
import pickle
from msrvtt_dataloader import MSRVTT_DataLoader, MSRVTT_TrainDataLoader
from lsmdc_dataloader import LSMDC_DataLoader
from epic_dataloader import Epic_DataLoader

import wandb
wandb.login()
wandb.init(project='howto100m_feature_context', entity='chefs')
config = wandb.config
config.clips_per_sample = 3
config.lr = 0.0001
args = get_args()
if args.verbose:
    print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
    os.mkdir(args.checkpoint_dir)

if not(args.youcook) and not(args.msrvtt) and not(args.lsmdc) and not(args.epic):
    print('Loading captions: {}'.format(args.caption_path))
    caption = pickle.load(open(args.caption_path, 'rb'))
    print('done')

print('Loading word vectors: {}'.format(args.word2vec_path))
we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
print('done')

if args.epic:
    root_path = '/raid/xiaoyuz1/EPIC'
    if args.epic_verb_only:
        gt_path = os.path.join(root_path, 'howto100m_groundTruth/verb')
    else:
        gt_path = os.path.join(root_path, 'howto100m_groundTruth/narration')
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
      start_idx = start_idx,
      gt_path = gt_path,
      we=we,
      we_dim=args.we_dim,
      max_words=args.max_words
    )
elif args.youcook:
    dataset = Youcook_DataLoader(
        data=args.youcook_train_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
elif args.msrvtt:
    dataset = MSRVTT_TrainDataLoader(
        csv_path=args.msrvtt_train_csv_path,
        json_path=args.msrvtt_train_json_path,
        features_path=args.msrvtt_train_features_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
elif args.lsmdc:
    dataset = LSMDC_DataLoader(
        csv_path=args.lsmdc_train_csv_path,
        features_path=args.lsmdc_train_features_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
    )
else:
    dataset = Youtube_DataLoader(
        csv=args.train_csv,
        features_path=args.features_path_2D,
        features_path_3D=args.features_path_3D,
        caption=caption,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        we=we,
        we_dim=args.we_dim,
        n_pair=args.n_pair,
    )
dataset_size = len(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_thread_reader,
    shuffle=True,
    batch_sampler=None,
    drop_last=True,
)
print(len(dataloader))

if args.eval_epic:
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
        subsample_csv=1000,
    )
    dataloader_lsmdc = DataLoader(
        dataset_lsmdc,
        batch_size=1000,
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
        batch_size=1000,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
net = Net(
    video_dim=args.feature_dim,
    embd_dim=args.embd_dim,
    we_dim=args.we_dim,
    n_pair=args.n_pair,
    max_words=args.max_words,
    sentence_dim=args.sentence_dim,
)
net.train()
# Optimizers + Loss
if args.epic and args.epic_verb_only:
    loss_op = TripletLoss(
        margin=args.margin,
        negative_weighting=args.negative_weighting,
        batch_size=args.batch_size,
        n_pair=args.n_pair,
        hard_negative_rate=args.hard_negative_rate,
    )
else:
    loss_op = MaxMarginRankingLoss(
        margin=args.margin,
        negative_weighting=args.negative_weighting,
        batch_size=args.batch_size,
        n_pair=args.n_pair,
        hard_negative_rate=args.hard_negative_rate,
    )
net.cuda()
loss_op.cuda()

if args.pretrain_path != '':
    args.pretrain_path = os.path.join('/raid/xiaoyuz1/EPIC', 'howto100m', 'model/howto100m_pt_model.pth')
    net.load_checkpoint(args.pretrain_path)

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.verbose:
    print('Starting training loop ...')

def TrainOneBatch(model, opt, data, loss_fun, epic=True):
    text = data['text'].cuda()
    video = data['video'].cuda()
    video = video.view(-1, video.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])
    opt.zero_grad()
    with th.set_grad_enabled(True):
        sim_matrix = model(video, text)
        if epic and args.epic_verb_only:
            labels = data['caption_cls'].cuda()
            loss = loss_fun(-sim_matrix, labels) * 1000
        else:
            loss = loss_fun(sim_matrix) * 100
    loss.backward()
    opt.step()
    return loss.item()

def Eval_retrieval(model, eval_dataloader, dataset_name, cnt, epic=False):
    interval = len(eval_dataloader) // 5
    model.eval()
    print('Evaluating Text-Video retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            text = data['text'].cuda()
            video = data['video'].cuda()
            m = model(video, text)
            m  = m.cpu().detach().numpy()
            if epic and args.epic_verb_only:
                metrics = compute_epic_metrics(m, data['caption_cls'].numpy().astype(int))
                metrics2 = compute_metrics(m.T)
                if (i_batch + 1) % interval == 0:
                    print_computed_metrics(metrics)
                    print_computed_metrics(metrics2)
            else:
                metrics = compute_metrics(m.T)
                r1 = metrics['R1']
                r5 = metrics['R5']
                r10 = metrics['R10']
                mr = metrics['MR']
                wandb_dict = {'val/R1': r1, 'val/R5': r5, 'val/R10':r10, 'val/MR':mr}
                wandb.log(wandb_dict,step=cnt)
                if (i_batch + 1) % interval == 0:
                    print_computed_metrics(metrics)

cnt = 0
for epoch in range(args.epochs):
    running_loss = 0.0
    if (epoch + 1) % args.eval_every == 0:
        if args.eval_youcook:
            Eval_retrieval(net, dataloader_val, 'YouCook2',cnt)
        if args.eval_msrvtt:
            Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT',cnt)
        if args.eval_lsmdc:
            Eval_retrieval(net, dataloader_lsmdc, 'LSMDC',cnt)
        if args.eval_epic:
            Eval_retrieval(net, dataloader_epic, 'EpicKitchens',cnt,epic=True)
    if args.verbose:
        print('Epoch: %d' % epoch)
    
    for i_batch, sample_batch in enumerate(dataloader):
        batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss_op, args.epic)
        wandb.log({'train/loss':batch_loss}, step=cnt)
        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose:
            print('Epoch %d, Epoch status: %.4f, Training loss: %.4f' %
            (epoch + 1, args.batch_size * float(i_batch) / dataset_size,
            running_loss / args.n_display))
            running_loss = 0.0
        cnt += 1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.lr_decay
    if args.checkpoint_dir != '' and (epoch+1)%10==0:
        path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(epoch + 1))
        net.save_checkpoint(path)

if args.eval_youcook:
    Eval_retrieval(net, dataloader_val, 'YouCook2',cnt)
if args.eval_msrvtt:
    Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT',cnt)
if args.eval_lsmdc:
    Eval_retrieval(net, dataloader_lsmdc, 'LSMDC',cnt)
if args.eval_epic:
    Eval_retrieval(net, dataloader_epic, 'EpicKitchens',cnt)
