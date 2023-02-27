import torch
import sys
import argparse
from torch.utils.data import DataLoader
from compressai.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from compressai.losses import RateDistortionLoss

from model import ELICModel
from train import AverageMeter
net = ELICModel()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Evaluation dataset"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--plot",action="store_true", default=True)
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = RateDistortionLoss()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net.update()
    net.eval()
    net.to(device)
    test_dataset = ImageFolder(args.dataset, split="test", transform=transforms.ToTensor())
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = net(d)

            r = net.compress(d)
            x_h = net.decompress(r['strings'],r['shape'])
            if args.plot:
                img = transforms.ToPILImage()(d.squeeze().cpu())
                rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())
                whole = transforms.ToPILImage()(x_h['x_hat'].squeeze().cpu())
                fix, axes = plt.subplots(1, 3, figsize=(16, 12))
                for ax in axes:
                    ax.axis('off')
                axes[0].imshow(img)
                axes[0].title.set_text('Original')
                axes[1].imshow(rec_net)
                axes[1].title.set_text('Reconstructed')
                axes[2].imshow(whole)
                axes[2].title.set_text('compress+decompress')
                plt.savefig("eval.png")


            out_criterion = criterion(out_net, d)
            aux_loss.update(net.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            

    print(
        f"Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
if __name__ == "__main__":
    main(sys.argv[1:])