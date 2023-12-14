import math, time, sys, os, torch

sys.path.append("/home/lutianen/final_design/")

from data import cifar10_dataset
from model.vgg_cifar import VGG
from utils.options import args
from utils import common as utils
from importlib import import_module


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

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir, utils.local_time, utils.local_time  + '.log'))
loss_func = torch.nn.CrossEntropyLoss()

# Data
print(">>> Preparing data...")
args.train_batch_size *= args.num_batches_per_step
data_loader = cifar10_dataset.Data(args)

# Load model
print(">>> Loading pretrained model...")
if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
    raise Exception("Pretrained model doesn't exist!")
ckpt = torch.load(args.pretrain_model, map_location=device)

if args.arch == "vgg_cifar":
    origin_model = import_module(f"model.{args.arch}").VGG(args.cfg).to(device)
origin_model.load_state_dict(ckpt['state_dict'])

def graphVGG(model, cr:float, v:dict) :
    indices = []
    centroids = {}
    grads = []
    index = 0

    # Get the grad of each layer(conv, fc) and put them into a list `grads`
    for idx, (name, item) in enumerate(model.named_parameters()):
        # Debug
        # print("self.model.named_parameters index: {} name: {} item.size(): {}".format(idx, name, item.size()))

        # Conv layer
        if "feature" in name and len(item.size()) == 4:
            lam = flops_lambda['vgg_cifar'] # hyper-parameter shared across the network
            numFLOPs = flops_cfg['vgg_cifar'][index] # FLOPs of the index-th layer

            convGrad = item.grad.data.view(item.grad.data.size(0), -1)
            convGrad = torch.norm(convGrad, 2, 1)
            grads.append(convGrad)

            preversedNum = math.ceil(convGrad.size(0) *(1 - cr))
            # preversedNum = int(convGrad.size(0) *(1 - cr))
            preversedGrad, _ = torch.topk(torch.abs(convGrad), preversedNum)
            threshold = preversedGrad[preversedNum-1]

            # TODO  验证 knn 方法选择的正确性
            originalConvGrad = item.grad.data

            cr = torch.sum(torch.lt(torch.abs(convGrad), threshold)).item()/convGrad.size(0)
            _, _, centroid, indice = utils.graphGrad(originalConvGrad, math.ceil(originalConvGrad.size(0) * (1 - cr)))
            centroids[name] = centroid.reshape(-1, originalConvGrad.size(1), originalConvGrad.size(2), originalConvGrad.size(3))
            indices.append(indice)

            index += 1
        # Full Connect layer
        # elif "classifier" in name and len(item.size()) == 2:
        #     classifierGrad = item.grad.data.view(item.grad.data.size(0), -1)
        #     classifierGrad = torch.norm(classifierGrad, 2, 1)
        #     # classifierGrad = classifierGrad.view(-1)
        #     grads.append(classifierGrad)

        #     preversedNum = math.ceil(classifierGrad.size(0) *(1 - cr))
        #     preversedGrad, _ = torch.topk(torch.abs(convGrad), preversedNum)
        #     threshold = preversedGrad[preversedNum-1]

        #     originalConvGrad = item.grad.data

        #     # cr = torch.sum(torch.lt(torch.abs(classifierGrad), threshold)).item()/classifierGrad.size(0)
        #     # _, _, centroid, indice = utils.graphGrad(originalConvGrad, int(originalConvGrad.size(0) * (1 - cr)))
        #     _, _, centroid, indice = utils.graphGrad(originalConvGrad, 
        #                                              math.ceil(int(originalConvGrad.size(0) * (1 - cr))))
        #     centroids[name] = centroid.reshape(-1, originalConvGrad.size(1))
        #     indices.append(indice)

    # 优化项：使用 set 避免双重循环
    centroidsKeys = set(centroids.keys())
    # Update the gradient with the centroids
    idx = -1
    for _, (name, item) in enumerate(model.named_parameters()):
        if name in centroidsKeys:
            idx += 1
            # if idx == 0: continue
            item.grad.data = utils.postProcessGrad(item.grad.data, torch.FloatTensor(centroids[name]), indices[idx], name, v)
    # print("GC finished.")

def train(model, optimizer, train_loader, args, epoch, v, topk=(1,)):
    model.train()
    losses = utils.AverageMeter('Time', ':6.3f')
    accurary = utils.AverageMeter('Time', ':6.3f')
    top5_accuracy = utils.AverageMeter('Time', ':6.3f')

    batch_size = int(args.train_batch_size / args.num_batches_per_step)
    step_size = args.num_batches_per_step * batch_size
    _r_num_batches_per_step = 1.0 / args.num_batches_per_step

    print_freq = len(train_loader.dataset) // batch_size // 10
    start_time = time.time()

    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        model = model.to(device)
        output = model(inputs[0: batch_size])
        loss = loss_func(output, targets[0: batch_size])
        loss.mul_(_r_num_batches_per_step)
        loss.backward()
        for b in range(batch_size, step_size, batch_size):
            _inputs = inputs[b:b+batch_size]
            _targets = targets[b:b+batch_size]
            _outputs = model(_inputs)
            _loss = loss_func(_outputs, _targets)
            _loss.mul_(_r_num_batches_per_step)
            _loss.backward()
            loss += _loss.item()
        
        # XXX GC
        graphVGG(model, args.pr_target, v)
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
