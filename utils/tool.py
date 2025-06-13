import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os

class MLRS(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """

    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        # self.diff = None
        if mode == 'train':
            self.data = [Image.open(os.path.join(root, 'MLRS', i)).convert('RGB') for i in MLRS.TRAIN_DATA]
            self.targets = MLRS.TRAIN_TARGETS
            # self.targets.dot(self.targets.T) == 0
        elif mode == 'query':
            self.data = [Image.open(os.path.join(root, 'MLRS', i)).convert('RGB') for i in MLRS.QUERY_DATA]
            self.targets = MLRS.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = [Image.open(os.path.join(root, 'MLRS', i)).convert('RGB') for i in
                         MLRS.RETRIEVAL_DATA]
            self.targets = MLRS.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)

    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train):
        # Load dataset
        img_txt_path = os.path.join(root, 'img.txt')
        targets_txt_path = os.path.join(root, 'targets.txt')

        # Read files
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(targets_txt_path, dtype=np.int64)

        # Split dataset
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(targets_txt_path, dtype=np.int64)

        # Split dataset
        perm_index = np.random.permutation(data.shape[0])
        query_index = perm_index[:num_query]
        train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query:]

        MLRS.QUERY_DATA = data[query_index]
        MLRS.QUERY_TARGETS = targets[query_index, :]

        MLRS.TRAIN_DATA = data[train_index]
        MLRS.TRAIN_TARGETS = targets[train_index, :]

        MLRS.RETRIEVAL_DATA = data[retrieval_index]
        MLRS.RETRIEVAL_TARGETS = targets[retrieval_index, :]


class ImageList(object):
    def __init__(self, image_list, labels=None, transform=None):
        self.imgs = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(open(path, 'rb')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.imgs)

# class ImageList(object):
#     def __init__(self, image_list, labels=None, transform=None):
#         self.imgs = [((Image.open(open(val.split()[0], 'rb')).convert('RGB')), np.array([int(la) for la in val.split()[1:]])) for val in image_list]
#         self.transform = transform
#
#     def __getitem__(self, index):
#         img, target = self.imgs[index]
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)
def rand_unit_sphere(npoints, ndim):
    '''
    Generates "npoints" number of vectors of size "ndim"
    such that each vectors is a point on an "ndim" dimensional sphere
    that is, so that each vector is of distance 1 from the center
    npoints -- number of feature vectors to generate
    ndim -- how many features per vector
    returns -- np array of shape (npoints, ndim), dtype=float64
    '''
    vec = np.random.randn(npoints, ndim)
    vec = np.divide(vec, np.expand_dims(np.linalg.norm(vec, axis=1), axis=1))
    return vec

def rand_unit_rect(npoints, ndim):
    '''
    Generates "npoints" number of vectors of size "ndim"
    such that each vectors is a point on an "ndim" dimensional sphere
    that is, so that each vector is of distance 1 from the center
    npoints -- number of feature vectors to generate
    ndim -- how many features per vector
    returns -- np array of shape (npoints, ndim), dtype=float64
    '''
    vec = np.random.randint(0, 2, size=(npoints, ndim))
    vec[vec==0] = -1
    return vec



# def get_dataloader(args):
#
#     transform = transforms.Compose([
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])])
#
#     train_image_path_txt = args.txtfile_path + args.dataset + "/train.txt"
#     test_image_path_txt = args.txtfile_path + args.dataset + "/test.txt"
#     database_image_path_txt = args.txtfile_path + args.dataset + "/database.txt"
#
#     train_dataset = ImageList("./data/UCMD/train/", open(train_image_path_txt).readlines(), transform)
#     test_dataset = ImageList("./data/UCMD/test/", open(test_image_path_txt).readlines(), transform)
#     database_dataset = ImageList("./data/UCMD/database/", open(database_image_path_txt).readlines(), transform)
#
#     train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
#     test_loader = util_data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
#     database_loader = util_data.DataLoader(database_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
#
#     return train_loader, test_loader, database_loader
#
# def compute_hashcode(dataloader, net):
#     bs, clses = [], []
#     net.eval()
#     for img, cls, _ in tqdm(dataloader):
#         clses.append(cls)
#         bs.append((net(img.cuda())).data.cpu())
#     return torch.cat(bs).sign(), torch.cat(clses)
#
# def compute_hash_center(trainloader, net, num_classes):
#     net.eval()
#     hash_center = []
#     data_dict = {}
#     for i in range(num_classes):
#         data_dict[i] = list()
#
#     for idx, (img, cls, _) in enumerate(trainloader):
#         fea = net(img.cuda())
#         fea = fea.data.cpu()
#
#         label = np.argmax(cls, axis=1)
#         for i in range(img.shape[0]):
#             data_dict[label[i].item()].append(fea[i].unsqueeze(0))
#
#     for i in range(num_classes):
#         nums = len(data_dict[i])
#         hash_center_i = torch.sum(torch.cat(data_dict[i]), dim=0) / nums
#         hash_center_i = hash_center_i.unsqueeze(0)
#         hash_center.append(hash_center_i)
#
#     return torch.cat(hash_center)
#
# def CalcHammingDist(B1, B2):
#     q = B2.shape[1]
#     distH = 0.5 * (q - np.dot(B1, B2.transpose()))
#     return distH
#
# def CalcMap(rB, qB, retrievalL, queryL):
#
#     num_query = queryL.shape[0]
#     map = 0
#     for iter in range(num_query):
#         gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
#         tsum = np.sum(gnd)
#         if tsum == 0:
#             continue
#         hamm = CalcHammingDist(qB[iter, :], rB)
#         ind = np.argsort(hamm)
#         gnd = gnd[ind]
#         count = np.linspace(1, tsum, int(tsum))
#
#         tindex = np.asarray(np.where(gnd == 1)) + 1.0
#         map_ = np.mean(count / (tindex))
#         # print(map_)
#         map = map + map_
#     map = map / num_query
#
#     return map

def load_preweights(model, preweights):
    # loading the pretrained weights
    state_dict = {}
    preweights = torch.load(preweights)
    train_parameters = model.state_dict()
    for pname, p in train_parameters.items():
        if pname == 'conv1.weight':
            state_dict[pname] = preweights["features.0.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv1.bias':
            state_dict[pname] = preweights["features.0.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv2.weight':
            state_dict[pname] = preweights["features.3.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv2.bias':
            state_dict[pname] = preweights["features.3.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv3.weight':
            state_dict[pname] = preweights["features.6.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv3.bias':
            state_dict[pname] = preweights["features.6.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv4.weight':
            state_dict[pname] = preweights["features.8.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv4.bias':
            state_dict[pname] = preweights["features.8.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv5.weight':
            state_dict[pname] = preweights["features.10.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'conv5.bias':
            state_dict[pname] = preweights["features.10.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'classifier.1.weight':
            state_dict[pname] = preweights["classifier.1.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'classifier.1.bias':
            state_dict[pname] = preweights["classifier.1.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'classifier.4.weight':
            state_dict[pname] = preweights["classifier.4.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'classifier.4.bias':
            state_dict[pname] = preweights["classifier.4.bias"]
            print("loading pretrained weights {}".format(pname))
        # elif pname == 'classifier.6.weight':
        #     state_dict[pname] = preweights["classifier.6.weight"]
        #     print("loading pretrained weights {}".format(pname))
        # elif pname == 'classifier.6.bias':
        #     state_dict[pname] = preweights["classifier.6.bias"]
        #     print("loading pretrained weights {}".format(pname))
        else:
            state_dict[pname] = train_parameters[pname]
    return state_dict