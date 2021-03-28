import torchvision.transforms as transforms

class MyConfig():
    def __init__(self, dic):
        self.num_classes = dic['num_classes']
        self.PATH = dic['PATH']
        self.PATH_FULL = dic['PATH_FULL']
        self.num_epochs = dic['num_epochs']
        self.number_shots = dic['number_shots']
        self.iteration = dic['iteration']
        self.Finetune = dic['Finetune']
        self.fs_classes = dic['fs_classes']
        self.fs_class = dic['fs_class']
        self.random = dic['random']
        self.batch_size = dic['batch_size']
        self.embedding_size = dic['embedding_size']
        self.CreateNets = dic['CreateNets']
        self.transform_cifar = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])