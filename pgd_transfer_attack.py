import torch
import argparse
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset 
# TODO: install following packages in Docker (update requirements.txt)
# from pytorchcv.model_provider import get_model as ptcv_get_model 
# from torchattacks import PGD
import torchvision.transforms as transform

#  clean: 89.15% & robust: 0.05% of base ResNet-20 on CIFAR-10 dataset
def create_attack():

    # dataset
    test_set = CIFAR10('.', train=False, download=True, transform=transform.ToTensor())
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # load pre-trained model
    model = ptcv_get_model('resnet20_cifar10', pretrained=True)  # 5.97 top-1 error
    model = model.eval().cuda()

    # prepare attack
    atck = PGD(model, eps=8/255, alpha=2/255, steps=10) # same params as in hydra
    atck.set_return_type('int') # Save as integer.
    atck.save(data_loader=test_loader, save_path="./cifar10_pgd.pt", verbose=True)


def test_loop(model, dataloader):
    model = model.eval()
    corr_pred = 0
    for images, labels in tqdm(dataloader):
        output = model(images.cuda())
        _, prediction = torch.max(output, 1)
        corr_pred += (labels.cuda() == prediction).sum()/ labels.size(0) # .detach().cpu().numpy()

    print(f"Robust (Black box) accuracy: {100 * corr_pred/len(dataloader):.2f}%")

def black_box_eval(model, adv_path="./cifar10_pgd.pt", batch_size=32):
    # Adversarial examples for attack
    adv_images, adv_labels = torch.load(adv_path)
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=False)
    print("Adversarial examples ready !!")

    # Clean dataset
    # test_set = CIFAR10('.', train=False, download=True, transform=transform.ToTensor())
    # adv_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # load pre-trained model
    # NOTE: support for multiple hydra model - done in HYDRA train script
    # model = ptcv_get_model('resnet20_cifar10', pretrained=True)  # 5.97 top-1 error
    # model = model.cuda()
    # print("Model ready !!")

    test_loop(model, adv_loader)

# def parse_args():
#     parser = argparse.ArgumentParser(description='Transfer (Black box) attack configs')
#     parser.add_argument('--mode', default='create', type=str, choices=('create', 'eval'), help='Choose between the two modes')
#     parser.add_argument('--path', default='./', help='Path to save or load model checkpoint')
#     return parser.parse_args()
        
if __name__ == '__main__':
    # args = parse_args()
    # if args.mode == 'create':
    # else:
        # black_box_eval() 
        
    create_attack()