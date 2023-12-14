import time, torch, os, math
import numpy as np
from data import cifar10_dataset
from utils.options import args
from utils import common as utils
from importlib import import_module

def getNowFormatTime() -> str:
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime(time.time()))

print(getNowFormatTime())


g = torch.rand((64, 3, 3, 3))
print(g.shape)
print(g.view(-1).shape)
print(g.view(g.size(0), -1).shape)


device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
# Load model
print(">>> Loading pretrained model...")
if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
    raise Exception("Pretrained model doesn't exist!")
ckpt = torch.load(args.pretrain_model, map_location=device)

if args.arch == "vgg_cifar":
    origin_model = import_module(f"model.{args.arch}").VGG(args.cfg).to(device)
origin_model.load_state_dict(ckpt['state_dict'])

# Data
print(">>> Preparing data...")
args.train_batch_size *= args.num_batches_per_step
data_loader = cifar10_dataset.Data(args)

start_epoch = 0
best_acc = 0.0
#test(origin_model,loader.testLoader, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))

print('==> Building Model..')
if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
    raise ('Pretrained_model path should be exist!')
if args.arch == 'vgg_cifar':
    # model, cfg = graph_vgg(args.pr_target)
    cfg = None
    model = import_module(f'model.{args.arch}').VGG(args.cfg, layer_cfg=cfg).to(device)
# elif args.arch == 'resnet_cifar':
    # model, cfg = graph_resnet(args.pr_target)
# elif args.arch == 'googlenet':
    # model, cfg = graph_googlenet(args.pr_target)
else:
    raise('arch not exist!')
print("Graph Down!")

if len(args.gpus) != 1:
    model = torch.nn.DataParallel(model, device_ids=args.gpus)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.lr_type == 'step':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
elif args.lr_type == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

# 初始化 v：用于梯度累积
v = {} 
for idx, (name, item) in enumerate(model.named_parameters()):
    if "feature" in name and len(item.size()) == 4:
        v[name] = torch.zeros_like(item)
    # Full Connect layer
    elif "classifier" in name and len(item.size()) == 2:
        v[name] = torch.zeros_like(item)

def sparsify_dgc(tensor, name):
    cr = 0.01
    numel = tensor.numel()
    shape = list(tensor.size())

    numSlects = int(math.ceil(numel * cr))
    tensor = tensor.view(-1) # 有四维张量拉伸成一维向量

    importance = tensor.abs()

    threshold = torch.min(torch.topk(importance, numSlects, 0, largest=True, sorted=False)[0])
    mask = torch.ge(importance, threshold)
    indices = mask.nonzero().view(-1)
    numIndices = indices.numel()

    indices = indices[: numSlects]
    values = tensor[indices]
    return values, indices, numel, shape, numSlects

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

def sparsify_knn(grad, name, cr):
    G = grad.clone() 
    numChannels, shape = G.size(0), list(G.size())

    numSelects = int(math.ceil(numChannels * cr))
    print(G.view(-1).size())
    print(G.view(G.size(0), -1).size())
    G = G.view(G.size(0), -1)
    s_matrix = torch.nn.functional.normalize(torch.exp(-pairwise_distances(G)), 1)

    # Sort
    sorted_value, indices = torch.sort(s_matrix, descending=True)
   
    if args.graph_gpu:
        indices = indices.cpu()
    indices = indices.numpy()

    k = numSelects # Calculate the nearest k channels of each channel
    m_tmp = 0 # Common nearest channels after k nearest neighbor channels intersect
    indicek = set()
    while m_tmp < numSelects:
        # Take the nearest k filters
        indicek = indices[:,:k].tolist()
        # Intersect k nearest neighbors of all filters
        indicek = set(indicek[0]).intersection(*indicek[1:])
        m_tmp = len(indicek)
        if m_tmp > numSelects:
            # Take the difference set for the result of the last KNN, 
            # and randomly select the filter from the difference set until the target m is reached
            pre_indicek = indices[:,:k-1].tolist()
            pre_indicek = set(pre_indicek[0]).intersection(*pre_indicek[1:])
            redundant_indice = indicek.difference(pre_indicek)
            while len(pre_indicek) != numSelects:
                pre_indicek.add(redundant_indice.pop())
            indicek = pre_indicek
            m_tmp = numSelects

        # logger.info('k[{}]\tm_tmp[{}]\ttarget m[{}]'.format(k,m_tmp,m))
        k += 1
    if args.graph_gpu:
        Gprune = torch.index_select(G,0,torch.tensor(list(indicek)).to(device))
    else:
        Gprune = torch.index_select(G,0,torch.tensor(list(indicek)))

    m_matrix = torch.nn.functional.normalize(torch.exp(-pairwise_distances(G, Gprune)),1)
    Gprune = Gprune.cpu()
    return m_matrix, s_matrix, Gprune, indicek


for name, param in model.named_parameters():
    if torch.is_tensor(param):
        numel = param.numel()
        shape = list(param.size())
        print(name, numel, shape)
        values, indices, numel, shape, num_selects = sparsify_dgc(param.data, name)
        sparsify_knn(param.data, name, 0.01)
        indices = indices.view(-1, 1)
        values = values.view(-1, 1)
    else:
        assert isinstance(param, (list, tuple))
        numel, shape = param[0], param[1]
        print(name, numel, shape)
