from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import logging
import time
import torch
import torch.nn as nn
from utils.options import args
import numpy as np
import torch.nn.functional as F
import random
from sklearn.cluster import KMeans
from sklearn.random_projection import SparseRandomProjection

import os

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

def getNowFormatTime(formatStr: str = "%Y.%m.%d-%H:%M:%S") -> str:
    return time.strftime(formatStr, time.localtime(time.time()))

local_time = getNowFormatTime()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

'''record configurations'''
class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)

        config_dir = os.path.join(self.job_dir, local_time, local_time + ' config.txt')
        #if not os.path.exists(config_dir):
        if args.resume != None:
            with open(config_dir, 'a') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        else:
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.args = args
        self.job_dir = Path(os.path.join(args.job_dir, local_time))
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        record_config(args)

    def save_model(self, state, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_last.pt'
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')



def graph_weight(weight, m, logger):
    '''
        weight: 原始模型的权重
        m: 需要保留的通道数
    '''

    if args.graph_gpu:
        W = weight.clone() 
    else:
        W = weight.cpu().clone()

    if weight.dim() == 4:  #Convolution layer
        W = W.view(W.size(0), -1)
    else:
        raise('The weight dim must be 4!')

    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

    # Calculate the similarity matrix and normalize
    s_matrix = F.normalize(torch.exp(-pairwise_distances(W)), 1)
    # Sort
    sorted_value, indices = torch.sort(s_matrix, descending=True)
   
    if args.graph_gpu: indices = indices.cpu()
    indices = indices.numpy()

    # Calculate the nearest k channels of each channel
    k = m 
    # Common nearest channels after k nearest neighbor channels intersect
    m_tmp = 0 
    while m_tmp < m:
        #Take the nearest k filters
        indicek = indices[:,:k].tolist()
        #Intersect k nearest neighbors of all filters
        indicek = set(indicek[0]).intersection(*indicek[1:])
        m_tmp = len(indicek)
        if m_tmp > m:
            #Take the difference set for the result of the last KNN, 
            #and randomly select the filter from the difference set until the target m is reached
            pre_indicek = indices[:,:k-1].tolist()
            pre_indicek = set(pre_indicek[0]).intersection(*pre_indicek[1:])
            redundant_indice = indicek.difference(pre_indicek)
            while len(pre_indicek) != m:
                pre_indicek.add(redundant_indice.pop())
            indicek = pre_indicek
            m_tmp = m

        #logger.info('k[{}]\tm_tmp[{}]\ttarget m[{}]'.format(k,m_tmp,m))
        k += 1
    if args.graph_gpu:
        Wprune = torch.index_select(W,0,torch.tensor(list(indicek)).to(device))
    else:
        Wprune = torch.index_select(W,0,torch.tensor(list(indicek)))
    m_matrix = F.normalize(torch.exp(-pairwise_distances(W,Wprune)),1)
    Wprune = Wprune.cpu()
    return m_matrix, s_matrix, Wprune, indicek

def kmeans_weight(weight,m,logger):
    W = weight.cpu().clone()
    filters_num = W.size(0)
    if weight.dim() == 4:  #Convolution layer
        W = W.view(filters_num, -1)
    else:
        raise('The weight dim must be 4!')
    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
    kmeans = KMeans(n_clusters=m, random_state=0).fit(W.numpy())
    indices = []
    if args.graph_gpu:
        m_matrix = F.normalize(torch.exp(-pairwise_distances(W.to(device),torch.from_numpy(kmeans.cluster_centers_).to(device))),1)
    else:
        m_matrix = F.normalize(torch.exp(-pairwise_distances(W,torch.from_numpy(kmeans.cluster_centers_))),1)
    indices = random.sample(range(0, W.size(0)-1), m)
    return m_matrix, torch.from_numpy(kmeans.cluster_centers_), indices


def random_weight(weight,m,logger):
    
    W = weight.cpu().clone()
    if weight.dim() == 4:  #Convolution layer
        W = W.view(W.size(0), -1)
    else:
        raise('The weight dim must be 4!')
    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
    indices = random.sample(range(0, W.size(0)-1), m)
    if args.graph_gpu:
        W = W.to(device)
        Wprune = torch.index_select(W,0,torch.tensor(list(indices)).to(device))
    else:
        Wprune = torch.index_select(W,0,torch.tensor(list(indices)))
    m_matrix = F.normalize(torch.exp(-pairwise_distances(W,Wprune)),1)
    return m_matrix, Wprune, indices

def getloss(B,A):
    #loss = torch.pow(torch.norm(A - torch.mm(B, B.t()),2),2)
    loss = torch.norm(A - torch.mm(B, B.t()),1)/(A.size(0)*A.size(0))
    '''
    f_num = A.size(0)
    A_1 = torch.zeros(f_num,f_num)
    for i in range(f_num):
        for j in range(f_num):
            A_1[i,j] = torch.sum(torch.mul(B[i,:],B[j,:]))
    loss = torch.pow(torch.norm(A - A_1,1),1)
    '''
    return loss


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def random_project(weight, channel_num):

    A = weight.cpu().clone()
    A = A.view(A.size(0), -1)
    rp = SparseRandomProjection(n_components=channel_num * weight.size(2) * weight.size(3))
    rp.fit(A)
    return rp.transform(A)

def direct_project(weight, indices):
    # print(weight.size())

    A = torch.randn(weight.size(0), len(indices), weight.size(2), weight.size(3))
    print(A.size())
    for i, indice in enumerate(indices):

        A[:, i, :, :] = weight[:, indice, :, :]

    return A

def postProcessGrad(grad, targetGrad, indices, name, v):
    '''
        梯度累积函数
    '''
    # print(grad.size())

    """
        A = torch.zeros(grad.size())
        A += grad
        index # 0.1
        c_grad = A[index] // 需要传输的梯度
        A[index] = 0 // 将已传输的梯度清零
        # orginal 的空洞梯度填充0
    """

    v[name] += grad # v 需要在外面初始化

    A = torch.zeros(grad.size()).to(grad.device)
    if v[name].dim() == 4: 
        for i, indice in enumerate(indices):
            # A[:, i, :, :] = A[:, indice, :, :]

            # A[:, i, :, :] = v[name][:, indice, :, :]
            # v[name][:, indice, :, :] = 0

            A[i, :, :, :] = v[name][indice, :, :, :]
            v[name][indice, :, :, :] = 0
    else:
        for i, indice in enumerate(indices):
            # A[:, i] = grad[:, indice]
            # A[:, i] = v[name][:, indice]
            # v[name][:, indice] = 0

            A[i,:] = v[name][indice, :]
            v[name][indice, :] = 0
    return A

#label smooth
class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def convert2Tensor(x):
    return torch.FloatTensor(x)

def getFilterGradCodebook(gradTorch, cr, modelLen, compuType: str = "norm2"):
    codebook = np.ones(modelLen)

    # conv
    if len(gradTorch.size()) == 4:
        filter_pruned_num = int(gradTorch.size()[0] * (1 - cr))
        gradVec = gradTorch.view(gradTorch.size()[0], -1)

        if compuType == "norm2":
            norm2 = torch.norm(gradVec, 2, 1).cpu().numpy()
        # elif compuType == "norm2With":
        #     lam = flops_lambda['vgg_cifar']
        #     numFLOPs = flops_cfg['vgg_cifar'][idx]
        #     norm2 = torch.div(gradTorch, math.pow(numFLOPs, lam))
        else:
            raise ValueError("compuType must be norm2 or norm2With")

        splitIdx = norm2.argsort()[: filter_pruned_num]

        kernelLen = gradTorch.size()[1] * gradTorch.size()[2] * gradTorch.size()[3]
        for x in range(0, len(splitIdx)):
            codebook[splitIdx[x] * kernelLen: (splitIdx[x] + 1) * kernelLen] = 0
    # Full connect
    elif len(gradTorch.size()) == 2:
        filter_pruned_num = int(gradTorch.size()[0] * (1 - cr))
        gradVec = gradTorch.view(gradTorch.size()[0], -1)

        if compuType == "norm2":
            norm2 = torch.norm(gradVec, 2, 1).cpu().numpy()
        # elif compuType == "norm2With":
        #     lam = flops_lambda['vgg_cifar']
        #     numFLOPs = flops_cfg['vgg_cifar'][idx]
        #     norm2 = torch.div(gradTorch, math.pow(numFLOPs, lam))
        else:
            raise ValueError("compuType must be norm2 or norm2With")

        splitIdx = norm2.argsort()[: filter_pruned_num]

        kernelLen = gradTorch.size()[1]
        for x in range(0, len(splitIdx)):
            codebook[splitIdx[x] * kernelLen: (splitIdx[x] + 1) * kernelLen] = 0
    else:
        pass
    return codebook


def initModelLength(model):
    modelSize, modelLength = {}, {}

    for index, item in enumerate(model.parameters()):
        modelSize[index] = item.size()

    for index1 in modelSize:
        for index2 in range(0, len(modelSize[index1])):
            if index2 == 0:
                modelLength[index1] = modelSize[index1][0]
            else:
                modelLength[index1] *= modelSize[index1][index2]
    return modelSize, modelLength

def graphGrad(grad, m):

    if args.graph_gpu:
        G = grad.clone() 
    else:
        G = grad.cpu().clone()

    # if grad.dim() == 4:  #Convolution layer
    G = G.view(G.size(0), -1)
    # else:
        # raise('The weight dim must be 4!')

    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
    # Calculate the similarity matrix and normalize

    # pairwise_distances -> GCC 中的距离函数
    temp = pairwise_distances(G)
    s_matrix = F.normalize(torch.exp(-temp), 1)

    # Sort
    sorted_value, indices = torch.sort(s_matrix, descending=True)
   
    if args.graph_gpu:
        indices = indices.cpu()
    indices = indices.numpy()

    k = m # Calculate the nearest k channels of each channel
    m_tmp = 0 # Common nearest channels after k nearest neighbor channels intersect
    indicek = set()
    while m_tmp < m:
        # Take the nearest k filters
        indicek = indices[:,:k].tolist()
        # Intersect k nearest neighbors of all filters
        indicek = set(indicek[0]).intersection(*indicek[1:])
        m_tmp = len(indicek)
        if m_tmp > m:
            # Take the difference set for the result of the last KNN, 
            # and randomly select the filter from the difference set until the target m is reached
            pre_indicek = indices[:,:k-1].tolist()
            pre_indicek = set(pre_indicek[0]).intersection(*pre_indicek[1:])
            redundant_indice = indicek.difference(pre_indicek)
            while len(pre_indicek) != m:
                pre_indicek.add(redundant_indice.pop())
            indicek = pre_indicek
            m_tmp = m

        # logger.info('k[{}]\tm_tmp[{}]\ttarget m[{}]'.format(k,m_tmp,m))
        k += 1
    if args.graph_gpu:
        Gprune = torch.index_select(G,0,torch.tensor(list(indicek)).to(device))
    else:
        Gprune = torch.index_select(G,0,torch.tensor(list(indicek)))

    m_matrix = F.normalize(torch.exp(-pairwise_distances(G, Gprune)),1)
    Gprune = Gprune.cpu()
    return m_matrix, s_matrix, Gprune, indicek
