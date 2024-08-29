from PIL import Image
from rough_sketch.utils.dataset import transform,CustomDataset
from torch.utils.data import DataLoader
from rough_sketch.net.network import UNet
from rough_sketch.utils.train import train
from rough_sketch.utils.visualization import loss_v
import torch
from rough_sketch.log.logger import logger

def main(ROOT = "./"):
    # Dataset Setting
    inputs = [Image.open(ROOT+f"/inputs/line/{str(i).zfill(3)}.jpg").convert('L') for i in range(1,101)]
    outputs = [Image.open(ROOT+f"/inputs/luf_line/{str(i).zfill(3)}_mask.png").convert('L') for i in range(1,101)]

    configs = {
        "transform":transform,
        "inputs_list":inputs,
        "outputs_list":outputs
    }
    dataset = CustomDataset(**configs)
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True)
    
    # Model Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device=device)
    optimizer = torch.optim.Adadelta(model.parameters())
    criterion = torch.nn.MSELoss().to(device=device)

    epochs = 1000
    logger.info("Training Start")

    configs = {
        "dataloader": dataloader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "device":device
    }

    for epoch in range(1,epochs+1):
        configs['epoch'] = epoch
        loss_train,total_loss_train = train(**configs)
        if epoch%100 == 0:
            logger.info(f"Save Model Epoch{epoch}")
            # model save
            torch.save(model.state_dict(),ROOT+f"rough_sketch/models/{epoch}_UNet.pth")

    # training result visialization 
    loss_v(ROOT=ROOT+"/rough_sketch/",total_loss_train=total_loss_train)


if __name__ == "__main__":
    main()