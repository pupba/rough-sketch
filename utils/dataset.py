import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

transform:transforms.transforms.Compose = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    # for trainning
    def __init__(self,
                 transform:transforms.transforms.Compose,
                 inputs_list:list[Image.Image],
                 outputs_list:list[Image.Image]):
        self.__transform = transform
        self.__inputs = inputs_list
        self.__outputs = outputs_list
        # preprocessing
        self.X,self.y = self.__preprocess()

    def __len__(self):
        return len(self.__inputs)

    def __preprocess(self)->tuple[list,list]:
        # inputs
        X = list(map(self.__transform,self.__inputs))
        # outputs
        y = list(map(self.__transform,self.__outputs))
        return X,y
    
    def __getitem__(self, index:int) -> tuple[torch.Tensor,torch.Tensor]:
        return self.X[index],self.y[index]
    
if __name__ == "__main__":
    ROOT = "/home/studiom/workspace2/undertone/"
    inputs = [Image.open(ROOT+f"/inputs/line/{str(i).zfill(3)}.jpg").convert('L') for i in range(1,101)]
    outputs = [Image.open(ROOT+f"/inputs/luf_line/{str(i).zfill(3)}_mask.png").convert('L') for i in range(1,101)]

    configs = {
        "transform":transform,
        "inputs_list":inputs,
        "outputs_list":outputs
    }
    dataset = CustomDataset(**configs)
    print(dataset.X[2].shape,dataset.y[2].shape)