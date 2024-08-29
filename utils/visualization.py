import matplotlib.pyplot as plt

def loss_v(ROOT:str,total_loss_train:list):
    fig = plt.figure()
    plt.plot(total_loss_train,label="training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT+"models/training_loss.png")

if __name__ == "__main__":
    loss_v("./rough-sketch/",[1,2,3,4,5,6,7])