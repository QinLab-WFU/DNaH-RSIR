from __future__ import print_function
import os
import sys
import argparse
import time
from utils.tool import *
from model import *
from torch.utils.data import DataLoader
import json
import centroids_generator
import scipy.io as scio
from timm.utils import AverageMeter
from loguru import logger




def parse_option():

    parser = argparse.ArgumentParser('argument for DNaH')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation (default: cuda:0)')
    parser.add_argument('--info', type=str, default='DAQH-CSAM+Ada4元组+(a = 1 b = 3 c = 1 wt_levels=2!)',
                        help='name')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='momentum (default: 0.9)')
    parser.add_argument('--betas', type=float, default=(0.9, 0.999),
                        help='betas (default: (0.9, 0.999))')
    parser.add_argument('--n_class', type=int, default=60,
                        help='number of class (default: UCMD:17 MLRS:60 DFC15:8)')
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument('--bit', type=int, choices=[16, 32, 48, 64, 128], default=128,
                        help='bit of hash code (choose from 16, 32, 48; default: 16)')
    parser.add_argument('--dataset', type=str, default='MLRS',###
                        help='remote sensing dataset')
    parser.add_argument('--pre_weight', default='/home/admin01/桌面/CXR/06-work/DAHNet-main/preweight/resnet50.pth',
                        help='path of pre-training weight of AlexNet')
    parser.add_argument('--root_path', default='/home/admin01/桌面/CXR/SCFR-main/data/MLRS/',
                        help='root directory where the dataset is placed')
    parser.add_argument('--save_path', default='',
                        help='path where the result is placed')
    parser.add_argument('--gpu', type=str, default='0',
                        help='selected gpu (default: 0)')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='scheduler (default: 0)')
    parser.add_argument('--step_size', type=int, default=80,
                        help='scheduler (default: 0)')
    parser.add_argument('--grama', type=float, default=0.1,
                        help='scheduler (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="random seed")

    parser.add_argument('--init-centroids-method', default='M', choices=['N', 'U', 'B', 'M', 'H'],
                        help='N = sign of gaussian; '
                             'B = bernoulli; '
                             'M = MaxHD'
                             'H = Hadamard matrix')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight of Hyp_loss (default:0.5)')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight of Hyp_loss (default:0.5)')

    parser.add_argument('--retrieve', type=int, default=0, help="retrieval number")

    parser.add_argument("--type_of_distance", type=str, default="cosine", help="cosine/euclidean/squared_euclidean")
    parser.add_argument("--type_of_quadruplets", type=str, default="all", help="all/semi-hard/hard")
    parser.add_argument("--what_is_hard", type=str, default="one", help="all/one")
    parser.add_argument("--epsilon", type=float, default=1,help="a(k)  hyper-parameter ε, aka. margin")
    parser.add_argument("--k_delta", type=float, default=3, help="b(r)  K_Δ of Eq. (7) in paper")

    parser.add_argument("--save_dir", type=str, default="./Output", help="directory to output results")
    args = parser.parse_args()
    return args



def prediction(loader, net):
    outputs = []
    labels = []

    for i, (images, lbls) in enumerate(loader):
        images = images.cuda()
        lbls = lbls.cuda()  # 将labels移到GPU
        b, local_f, cls, cls1, cls2, cls3 = net(images)
        outputs.append(b.detach().cpu())
        labels.append(lbls.detach().cpu())

    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs.numpy(), labels.numpy()

def _dataset(dataset,retrieve):
    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    if dataset == 'AID':
        if retrieve == 0:
            retrieve = 3000
        num_classes = 17
        trainset = ImageList(open('./data/AID/train.txt', 'r').readlines(), transform=data_transform['train'])
        testset = ImageList(open('./data/AID/test.txt', 'r').readlines(), transform=data_transform['val'])
        database = ImageList(open('./data/AID/database.txt', 'r').readlines(), transform=data_transform['val'])

    elif dataset == 'UCMD':
        if retrieve == 0:
            retrieve = 2100
        num_classes = 17
        trainset = ImageList(open('./data/UCMD/train.txt', 'r').readlines(), transform=data_transform['train'])
        testset = ImageList(open('./data/UCMD/test.txt', 'r').readlines(), transform=data_transform['val'])
        database = ImageList(open('./data/UCMD/database.txt', 'r').readlines(), transform=data_transform['val'])

    elif dataset == 'DFC15':
        if retrieve == 0:
            retrieve = 3342
        num_classes = 8
        trainset = ImageList(open('./data/DFC15/train.txt', 'r').readlines(), transform=data_transform['train'])
        testset = ImageList(open('./data/DFC15/test.txt', 'r').readlines(), transform=data_transform['val'])
        database = ImageList(open('./data/DFC15/database.txt', 'r').readlines(), transform=data_transform['val'])
    if dataset == 'MLRS':
        num_classes = 60
        if retrieve == 0:
            retrieve = 1000
        MLRS.init('./data/MLRS/', 1000, 5000)
        trainset = MLRS('./data/', 'train', transform=data_transform['train'])
        testset = MLRS('./data/', 'query', transform=data_transform['val'])
        database = MLRS('./data/', 'retrieval', transform=data_transform['val'])
    num_train, num_test,num_database = len(trainset),len(testset),len(database)

    dsets = (trainset,testset,database)
    nums = (num_train, num_test,num_database)
    return nums, dsets,retrieve, num_classes

def main():

    args = parse_option()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.add(
        'logs/{time}' + args.info + '_' + args.dataset+ '.log',
        rotation='50 MB',
        level='DEBUG'
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_flag = True
    save_flag = True
    path = './Result/' + args.dataset + '_' + args.info + '_' + str(args.bit) +'bits'
    if train_flag and save_flag:
        file_path = path + '.txt'
        f = open(file_path, 'w')
    nums, dsets, retrieval, class_num = _dataset(args.dataset, args.retrieve)
    num_train, num_test, num_database = nums
    dset_train, dset_test, dset_database = dsets

    net = DAHNET.dahnet(code_length=args.bit, num_classes=args.n_class, feat_size=2048,
                           device=args.device, pretrained=True)

    net.cuda()



    from loss.AdaQuadrupletLoss import AdaQuadrupletLoss
    criterion = AdaQuadrupletLoss(args, need_cnt=True)
    weight = load_preweights(net, preweights=args.pre_weight)
    net.load_state_dict(weight)

    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': args.learning_rate}
                                 # {'params': net.get_hash_params(), 'lr': 0.0001},
                                 ],
                                 # momentum=0.9,
                                 betas=(0.9, 0.999),
                                 weight_decay=0.0005,eps=1e-4)

    scheduler = centroids_generator.scheduler(args.scheduler,args.step_size,args.grama, optimizer)

    best_map = 0  # best map
    best_epoch = 0  # best epoch
    total_time = 0
    total_loss = 0
    train_loss = []
    mAPs = []
    Time = []

    quadruplet_meter = AverageMeter()
    loss_meter = AverageMeter()

    cross = nn.CrossEntropyLoss()
    cross_loss = AverageMeter()

    for epoch in range(args.epochs):
        net.train()
        trainloader = DataLoader(dataset=dset_train,
                                 batch_size=64,
                                 shuffle=True,
                                 num_workers=0)

        for i, (images, labels) in enumerate(trainloader):
            start = time.time()
            optimizer.zero_grad()
            batch_x = images.cuda()
            batch_y = labels.cuda()


            net = net.to(device)
            images = images.to(device)
            labels = labels.float()
            labels = labels.to(device)

            y_hat, local_f, cls, cls1, cls2, cls3 = net(images)

            cls_loss = (1.0 / 2.0) * cross(cls, labels) + \
                        (1.0 / 6.0) * (cross(cls1, labels) + cross(cls2, labels) + cross(cls3, labels))

            # 自适应四元组
            loss, n_quadruplets = criterion(y_hat, labels)
            quadruplet_meter.update(n_quadruplets)
            if n_quadruplets == 0:
                continue
            loss_meter.update(loss.item())
            c = 1
            Loss = loss + c*cls_loss
            loss_meter.update(Loss.item())
            optimizer.zero_grad()

            Loss.backward()
            optimizer.step()


            total_time += time.time() - start
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] \t Dataset：%s \t Bit: %d \t\tLoss: %.4f  \t Time: %.4f  '
                             % (epoch, args.epochs, i, num_train, args.dataset,args.bit,Loss.item(),total_time))
            # print(proxies)
            if save_flag:
                f.write('| Epoch [' +
                        str(epoch) +
                        '/' +
                        str(args.epochs) +
                        '] Iter['
                        + str(i) +
                        '/' +
                        str(num_train) +
                        'Dataset:' +
                        str(args.dataset)+
                        'Bit:'
                        +str(args.bit)+
                        '] total_loss:' +
                        str(Loss.item()) +
                        '] Time:'
                        + str(total_time) +
                        '\n')
        scheduler.step()

        '''
                training procedure finishes, evaluation
                '''

        net.eval()
        # save_dir = '/home/admin01/桌面/论文代码/SCFR-main/PR_curve/28/MLRS/'
        # os.makedirs(save_dir, exist_ok=True)
        save_model = '' + args.dataset +str(args.bit)
        os.makedirs(save_model, exist_ok=True)
        testloader = DataLoader(dataset=dset_test,
                                batch_size=64,
                                shuffle=False,
                                num_workers=0)

        databaseloader = DataLoader(dataset=dset_database,
                                    batch_size=64,
                                    shuffle=False,
                                    num_workers=0)
        if train_flag or not os.path.exists(path + '.data'):
            with torch.no_grad():
                data_predict, data_label = prediction(databaseloader, net)
                test_predict, test_label = prediction(testloader, net)

            if not train_flag and save_flag:
                datafile = open(path + '.data', 'w')
                datafile.write(
                    json.dumps(
                        [data_predict.tolist(), data_label.tolist(), test_predict.tolist(), test_label.tolist()]))
                datafile.close()
                print('------------- save data -------------')
        else:
            datafile = open(path + '.data', 'r').read()
            data = json.loads(datafile)
            data_predict = np.array(data[0])
            data_label = np.array(data[1])
            test_predict = np.array(data[2])
            test_label = np.array(data[3])
            print('------------- load data -------------')


        data_predict = np.sign(data_predict)
        test_predict = np.sign(test_predict)
        similarity = 1 - np.dot(test_predict, data_predict.T) / args.bit
        sim_ord = np.argsort(similarity, axis=1)

        apall = np.zeros(num_test)
        for i in range(num_test):
            x = 0
            p = 0
            order = sim_ord[i]
            for j in range(retrieval):
                if np.dot(test_label[i], data_label[order[j]]) > 0:
                    x += 1
                    p += float(x) / (j + 1)
            if p > 0:
                apall[i] = p / x
        mAP = np.mean(apall)
        if mAP > best_map:
            best_map = mAP
            best_epoch = epoch
            print("epoch: ", epoch)
            print("best_map: ", best_map)
            if save_flag:
                f.write("epoch: " + str(epoch) + '\n')
                f.write("best_map: " + str(best_map) + '\n')
                # torch.save(net.state_dict(), os.path.join(save_model, "-" + str(best_map) +"-"+ str(epoch) + ".pth"))
        else:
            print("epoch: ", epoch)
            print("map: ", mAP)
            print("best_epoch: ", best_epoch)
            print("best_map: ", best_map)
            if save_flag:
                f.write("epoch: " + str(epoch) + '\n')
                f.write("map: " + str(mAP) + '\n')
                f.write("best_epoch: " + str(best_epoch) + '\n')
                f.write("best_map: " + str(best_map) + '\n')

        retrieval_img = data_predict
        retrieval_labels = data_label
        query_img = test_predict
        query_labels = test_label

        result_dict = {
            'q_img': query_img,
            'r_img': retrieval_img,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(
            os.path.join(args.save_dir, args.info + args.dataset+"_"+str(best_map) + "_"+str(args.bit) + ".mat"),
            result_dict)
if __name__ == "__main__":

    main()
