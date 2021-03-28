import torch
import torch.nn as nn
import yaml
from config.conf_class import MyConfig

# %% Load config
with open(r'./config.yaml') as file:
    cfg_dict = yaml.load(file, Loader=yaml.FullLoader)

cfg = MyConfig(cfg_dict)


PATH = cfg.PATH

# Inspired from https://github.com/YU1ut/imprinted-weights
class Net(nn.Module):
    def __init__(self, num_classes, norm=True, scale=True):
        super(Net,self).__init__()
        self.norm = norm
        self.scale = scale
        self.extractor = Extractor()
        self.embedding = Embedding()
        self.classifier = Classifier(num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        if self.norm:
            x = self.l2_norm(x)
        if self.scale:
            x = self.s * x
        x = self.classifier(x)
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = self.l2_norm(x)
        return x

    def l2_norm(self,x):
        x_size = x.size()
        temp = torch.pow(x, 2)
        sum_x = torch.sum(temp, 1).add_(1e-10)
        norm = torch.sqrt(sum_x)
        x_out = torch.div(x, norm.view(-1, 1).expand_as(x))
        x_out = x_out.view(x_size)

        return x_out
    
    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))
    

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor,self).__init__()
        basenet = torch.load(PATH)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding,self).__init__()
        basenet = torch.load(PATH)
        fc_infeatures = basenet.fc.in_features
        self.fc = nn.Linear(fc_infeatures, cfg.embedding_size, bias=False)
    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(cfg.embedding_size, num_classes, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x