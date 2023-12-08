import math, time, sys, os, torch

sys.path.append("/home/tianen/doc/_XiDian/___FinalDesign/FinalDesign/final_design")

from data import cifar10_dataset

from model.vgg_cifar import VGG
from utils.options import args
from utils import common as utils

from importlib import import_module

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir, 'logger.log'))
loss_func = torch.nn.CrossEntropyLoss()

flops_cfg = {
    'vgg_cifar':[1.0, 18.15625, 9.07812, 18.07812, 9.03906, 18.03906, 18.03906, 9.01953, 18.01953, 18.01953, 4.50488, 4.50488, 4.50488],
    'resnet56':[1.0, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.67278, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.66667, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685],
    'resnet110':[1.0, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.9052, 0.67278, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.89297, 0.66667, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685, 0.88685],
    'mobilenet_v2':[1,3,1.5,0.5,2,1.5,1,0.5]
}
flops_lambda = {
 'vgg_cifar': 0.5,
 'resnet56':10,
 'resnet110':5,
 'mobilenet_v2':1
}

# Data
print(">>> Preparing data...")
data_loader = cifar10_dataset.Data(args)

# Load model
print(">>> Loading pretrained model...")
if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
    raise Exception("Pretrained model doesn't exist!")
ckpt = torch.load(args.pretrain_model, map_location=device)

if args.arch == "vgg_cifar":
    origin_model = import_module(f"model.{args.arch}").VGG(args.cfg).to(device)
elif args.arch == "resnet_cifar":
    origin_model = import_module(f"model.{args.arch}").ResNet(args.cfg).to(device)
# ...

origin_model.load_state_dict(ckpt['state_dict'])

# 算法实现
def graph_vgg(pr_target):    

    weights = []

    cfg = []
    pr_cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    current_layer = 0
    index = 0
    #start_time = time.time()
    #Sort the weights and get the pruning threshold
    for name, module in origin_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # conv_weight = module.weight.data
            # torch.div(): 用于对两个张量或一个张量和一个标量进行元素级除法

            # Eq.(4) 
            lam = flops_lambda['vgg_cifar'] # hyper-parameter shared across the network
            numFLOPs = flops_cfg['vgg_cifar'][index] # FLOPs of the index-th layer
            conv_weight = torch.div(module.weight.data, math.pow(numFLOPs, lam))
            weights.append(conv_weight.view(-1))
            index += 1

    all_weights = torch.cat(weights,0)
    preserve_num = int(all_weights.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
    threshold = preserve_weight[preserve_num-1]

    # Based on the pruning threshold, the prune cfg of each layer is obtained
    for weight in weights:
        pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
    # print(pr_cfg)

    
    # Get the preseverd filters after pruning by graph method based on pruning proportion
    for name, module in origin_model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            conv_weight = module.weight.data
            print(current_layer)
            if args.graph_method == 'knn':
                _, _, centroids, indice = utils.graph_weight(conv_weight, int(conv_weight.size(0) * (1 - pr_cfg[current_layer])), logger)
            elif args.graph_method == 'kmeans':
                _, centroids, indice = utils.kmeans_weight(conv_weight, int(conv_weight.size(0) * (1 - pr_cfg[current_layer])),logger)
            elif args.graph_method == 'random':
                _, centroids, indice = utils.random_weight(conv_weight, int(conv_weight.size(0) * (1 - pr_cfg[current_layer])),logger)
            else:
                raise('Method not exist!')
            cfg.append(len(centroids))
            indices.append(indice)

            # print("conv_weight.size: ", conv_weight.size())
            centroids_state_dict[name + '.weight'] = centroids.reshape((-1, conv_weight.size(1), conv_weight.size(2), conv_weight.size(3)))
            # print(": ", centroids_state_dict[name + '.weight'].size())

            prune_state_dict.append(name + '.bias')
            current_layer+=1

        elif isinstance(module, torch.nn.BatchNorm2d):
            prune_state_dict.append(name + '.weight')
            prune_state_dict.append(name + '.bias')
            prune_state_dict.append(name + '.running_var')
            prune_state_dict.append(name + '.running_mean')

        elif isinstance(module, torch.nn.Linear):
            prune_state_dict.append(name + '.weight')
            prune_state_dict.append(name + '.bias')

    #load weight
    model = import_module(f'model.{args.arch}').VGG(args.cfg, layer_cfg=cfg).to(device)

    if args.init_method == 'random_project' or args.init_method == 'direct_project':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())


        for i, (k, v) in enumerate(centroids_state_dict.items()):
            if i == 0: #first conv need not to prune channel
                continue
            if args.init_method == 'random_project':
                centroids_state_dict[k] = utils.random_project(torch.FloatTensor(centroids_state_dict[k]),
                                                         len(indices[i - 1]))
            else:
                centroids_state_dict[k] = utils.direct_project(torch.FloatTensor(centroids_state_dict[k]), indices[i - 1])

        for k, v in state_dict.items():
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
            else:
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg


def graphVGG(model, cr:float, v) :
    cfg = []
    indices = []
    centroids = {}

    maskIndex = []
    grads = []
    crCfg = []

    # Get the grad of each layer(conv, fc) and put them into a list `grads`
    for idx, (name, item) in enumerate(model.named_parameters()):
        # Debug
        # print("self.model.named_parameters index: {} name: {} item.size(): {}".format(idx, name, item.size()))

        # Conv layer
        if "feature" in name and len(item.size()) == 4:
            convGrad = torch.norm(item.grad.data, 2, 1) # 卷积层梯度的 L2 范数，并在输出的通道维度上进行归一化
            convGrad = convGrad.view(-1)
            grads.append(convGrad)

            preversedNum = int(convGrad.size(0) *(1 - cr))
            preversedGrad, _ = torch.topk(torch.abs(convGrad), preversedNum)
            threshold = preversedGrad[preversedNum-1]

            # TODO  验证 knn 方法选择的正确性

            cr = torch.sum(torch.lt(torch.abs(convGrad), threshold)).item()/convGrad.size(0)
            _, _, centroid, indice = utils.graphGrad(orginalConvGrad, int(orginalConvGrad.size(0) * (1 - cr)))
            centroids[name] = centroid.reshape(-1, orginalConvGrad.size(1), orginalConvGrad.size(2), orginalConvGrad.size(3))
            indices.append(indice)

        # Full Connect layer
        elif "classifier" in name and len(item.size()) == 2:
            classifierGrad = torch.norm(item.grad.data.view(item.grad.size(0), -1), 2, 1)
            classifierGrad = classifierGrad.view(-1)
            grads.append(classifierGrad)

            preversedNum = int(classifierGrad.size(0) *(1 - cr))
            preversedGrad, _ = torch.topk(torch.abs(convGrad), preversedNum)
            threshold = preversedGrad[preversedNum-1]

            orginalConvGrad = item.grad.data

            cr = torch.sum(torch.lt(torch.abs(classifierGrad), threshold)).item()/classifierGrad.size(0)
            _, _, centroid, indice = utils.graphGrad(orginalConvGrad, int(orginalConvGrad.size(0) * (1 - cr)))
            centroids[name] = centroid.reshape(-1, orginalConvGrad.size(1))
            indices.append(indice)

    # 优化项：使用 set 避免双重循环
    centroidsKeys = set(centroids.keys())
    # Update the gradient with the centroids
    idx = -1
    for _, (name, item) in enumerate(model.named_parameters()):
        if name in centroidsKeys:
            idx += 1
            if idx == 0: continue
            item.grad.data = utils.postProcessGrad(item.grad.data, torch.FloatTensor(centroids[name]), indices[idx - 1], name, v)

def train(model, optimizer, train_loader, args, epoch, v, topk=(1,)):
    model.train()
    losses = utils.AverageMeter('Time', ':6.3f')
    accurary = utils.AverageMeter('Time', ':6.3f')
    top5_accuracy = utils.AverageMeter('Time', ':6.3f')
    print_freq = len(train_loader.dataset) // args.train_batch_size // 10
    start_time = time.time()

    # 

    for batch, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        
        # XXX GC
        graphVGG(model, args.pr_target, v)
        model = model.to(device)

        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accurary.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Accuracy {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(train_loader.dataset),
                        float(losses.avg), float(accurary.avg), cost_time
                    )
                )
            else:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(train_loader.dataset),
                        float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), cost_time
                    )
                )
            start_time = current_time

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter('Time', ':6.3f')
    accurary = utils.AverageMeter('Time', ':6.3f')
    top5_accuracy = utils.AverageMeter('Time', ':6.3f')

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accurary.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            logger.info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
            )
        else:
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return accurary.avg
    else:
        return top5_accuracy.avg

def main():
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

    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, data_loader.train_loader, args, epoch, v, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))
        scheduler.step()
        test_acc = test(model, data_loader.test_loader, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
            'cfg': cfg
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accuracy: {:.3f}'.format(float(best_acc)))



if __name__ == '__main__':
    main()
