from PIL import Image
from rough_sketch.utils.dataset import transform
from rough_sketch.net.network import UNet
import torch

def load_model(model_name:str="./rough_sketch/models/1000_UNet.pth",device:torch.device = torch.device("cpu"))->UNet:
    model = UNet()
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()
    return model    

def inference(img:Image,device:torch.device = torch.device("cpu"))->Image.Image:
    # Resize 1024,1024
    # Image To Tensor
    original_size = img.size
    X = transform(img).unsqueeze(0) # add batch dim
    # Model predict
    model = load_model(device=device)
    X.to(device)
    with torch.no_grad():
        result = model(X)
    # Convert PIL Image
    result_img = result.squeeze(0).cpu().numpy() # remove batch dim
    result_img = (result_img * 255).astype("uint8")[0] # scaling
    
    image = Image.fromarray(result_img,mode="L")
    image = image.resize(original_size, Image.BILINEAR)
    image.save("test_result.png")
    return image

if __name__ == "__main__":
    img = Image.open("./test_line.png").convert('L')
    # img = Image.open("inputs/line/001.jpg").convert('L')
    inference(img)
