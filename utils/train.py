from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
from rough_sketch.log.logger import logger

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum+= val * n
        self.count += n
        self.avg = self.sum / self.count

def train(dataloader:DataLoader,model:torch.nn.Module,criterion:torch.nn.MSELoss,optimizer:torch.optim.Adadelta,epoch:int,device:torch.device=torch.device("cuda")):
    # training mode
    model.train()

    train_loss = AverageMeter()

    curr_iter = (epoch - 1) * len(dataloader)
    total_loss_train = []
    for i,data in enumerate(dataloader):
        images,targets = data
        X = Variable(images).to(device=device)
        Y = Variable(targets).to(device=device)
        optimizer.zero_grad() # grad init
        result = model(X) # model predict
        
        # Save sample        
        comb = torch.cat([X,result,Y],dim=0)
        save_image(comb,f"./rough_sketch/models/samples/epoch{epoch}_{i+1}iter_sample.png")

        loss = criterion(result,Y) # loss
        loss.backward() # back forward
        optimizer.step() # weight update
        train_loss.update(loss.item()) # loss_update
        curr_iter +=1 # iter
        # print log
        logger.info(f"Epoch{epoch}\tIter: {i + 1}/{len(dataloader)}\tTrain Loss Avg: {train_loss.avg:.4f}")
        total_loss_train.append(train_loss.avg)

    logger.info(f"\n\nEpoch{epoch} is Finished")
    return train_loss.avg,total_loss_train