# python eval_mode.py checkpoint  --checkpoint "./checkpoint_best_loss.pth.tar" --dataset "./CLIC Dataset/" -d "./output_eval_clic"
# python eval_mode.py checkpoint  --checkpoint "./checkpoint_best_loss.pth.tar" --dataset "./KODAK Dataset/" -d "./output_eval_kodak"

# python plot.py -f './output_eval_kodak/ELIC MODEL.json' -t "ELIC Model Evaluation on KODAK dataset [PSNR]" -o './output_eval_kodak/elic_kodak_psnr.png' --show
# python plot.py -f './output_eval_kodak/ELIC MODEL.json' -t "ELIC Model Evaluation on KODAK dataset [MS-SSIM]" -o './output_eval_kodak/elic_kodak_mssim.png' -m 'ms-ssim' --show

# python plot.py -f './output_eval_kodak/ELIC MODEL.json' -t "ELIC Model Evaluation on CLIC dataset [PSNR]" -o './output_eval_kodak/elic_clic_psnr.png' --show
# python plot.py -f './output_eval_kodak/ELIC MODEL.json' -t "ELIC Model Evaluation on CLIC dataset [MS-SSIM]" -o './output_eval_kodak/elic_clic_msssim.png' -m 'ms-ssim' --show

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt

import compressai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.ops import compute_padding
from compressai.zoo import image_models as pretrained_models
from compressai.zoo.image import model_architectures as architectures

from PIL import Image
from pytorch_msssim import ms_ssim
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

from model import ELICModel
from train import AverageMeter


def custom_collate(batch):
    
    # iterate over the batch and resize to the closest power of 2 (same for height and width)
    max_dim = 0
    for i in range(len(batch)):
        
        # get the image
        img = batch[i] # (3, h, w)

        # get the closest power of 2 for the height and width
        h = int(2**np.floor(np.log2(img.shape[1])))
        w = int(2**np.floor(np.log2(img.shape[2])))
        
        max_dim = max(h,w, max_dim)
        
        # resize the image keeping the aspect ratio with skimage
        img = resize(img.numpy(), (3, max_dim, max_dim), preserve_range=True, anti_aliasing=True)
        # add the image to the batch
        batch[i] = torch.from_numpy(img)
        
    # stack the images
    batch = torch.stack(batch)
    
    return batch


# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr-rgb"] = psnr(org, rec).item()
    metrics["ms-ssim-rgb"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def read_image(filepath: str) -> torch.Tensor:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, test_dataloader, criterion, device, loss, bpp_loss, mse_loss, aux_loss, output_dir):
    

    offset = 0
    metrics = defaultdict(float)
    metrics_list = defaultdict(list)
    
    for test in test_dataloader:
        tqdm.write(f"Processing batches {offset // test.shape[0]}")

        for i in tqdm(range(test.shape[0])):

            x = test[i]
            x = x.unsqueeze(0)

            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            
            x_padded = F.pad(x, pad, mode="constant", value=0)
            
            x = x.to(device)
            out_net = model(x)

            start = time.time()
            out_enc = model.compress(x_padded)
            enc_time = time.time() - start
            
            start = time.time()
            out_dec = model.decompress(out_enc['strings'],out_enc['shape'])
            dec_time = time.time() - start
            
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

            # input images are 8bit RGB for now
            _metrics = compute_metrics(x, out_dec["x_hat"], 255)
            num_pixels = x.size(0) * x.size(2) * x.size(3)
            bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            
            print(f'bpp: {bpp}')
            
            # iterate over the batch and plot the images
            img = transforms.ToPILImage()(x.squeeze().cpu())
            rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())
            whole = transforms.ToPILImage()(out_dec['x_hat'].squeeze().cpu())
            
            imgs = [img, rec_net, whole]
            titles = ['Original', 'Forward pass', 'Compression + Decompression']
            # save with suplot iteratively
            _, axs = plt.subplots(1, len(imgs), figsize=(10, 5))
            plt.suptitle(f'ELIC Result')
            for ix,_ in enumerate(imgs):
                axs[ix].imshow(imgs[ix])
                axs[ix].set_title(titles[ix])
                axs[ix].axis('off')
            plt.tight_layout()
            plt.savefig(f'./{output_dir}/{offset+i}.png')
                        

            out_criterion = criterion(out_net, x)
            aux_loss.update(model.aux_loss())
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
            
            rv = {
                "psnr-rgb": _metrics["psnr-rgb"],
                "ms-ssim-rgb": _metrics["ms-ssim-rgb"],
                "bpp": bpp,
                "encoding_time": enc_time,
                "decoding_time": dec_time,
            }
            for k, v in rv.items():
                metrics[k] += v
                metrics_list[k].append(v)
                

        offset += test.shape[0]
        


    for k, v in metrics.items():
        metrics[k] = v / len(test_dataloader)
        metrics_list[k] = sorted(list(np.array(metrics_list[k])/ len(test_dataloader)))
        
        
    return metrics, metrics_list
        


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True, progress=False
    ).eval()


def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    # update model if need be
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    # compatibility with 'not updated yet' trained nets
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]

    model_cls = architectures[arch]
    net = model_cls.from_state_dict(state_dict)
    if not no_update:
        net.update(force=True)
    return net.eval()


def eval_model(
    model: nn.Module,
    output_dir: Path,
    test_dataloader, 
    criterion, 
    device, 

    **args: Any,
) -> Dict[str, Any]:
    # device = next(model.parameters()).device
    metrics = defaultdict(float)
        
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    metrics, metrics_list = inference(model, test_dataloader, criterion, device, loss, bpp_loss, mse_loss, aux_loss, output_dir)
        
    results = defaultdict(list)
    
    for k, v in metrics.items():
        results[k].append(v)

        
        
    return results, metrics_list


def setup_args():
    # Common options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    # parent_parser.add_argument(
    #     "-a",
    #     "--architecture",
    #     type=str,
    #     choices=pretrained_models.keys(),
    #     help="model architecture",
    #     required=True,
    # )
    # parent_parser.add_argument(
    #     "-c",
    #     "--entropy-coder",
    #     choices=compressai.available_entropy_coders(),
    #     default=compressai.available_entropy_coders()[0],
    #     help="entropy coder (default: %(default)s)",
    # )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--checkpoint",
        help="Checkpoint tar path",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    # parent_parser.add_argument(
    #     "--entropy-estimation",
    #     action="store_true",
    #     help="use evaluated entropy estimation (no entropy coding)",
    # )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parent_parser.add_argument(
        "-d",
        "--output_directory",
        type=str,
        default="",
        help="path of output directory. Optional, required for output json file, results per image. Default will just print the output results.",
    )
    parent_parser.add_argument(
        "-of",
        "--output-file",
        type=str,
        default="",
        help="output json file name, (default: architecture-entropy_coder.json)",
    )
    parent_parser.add_argument(
        "--per-image",
        action="store_true",
        help="store results for each image of the dataset, separately",
    )
    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        type=str,
        default="1",
        help="Pretrained model qualities. (example: '1,2,3,4') (default: %(default)s)",
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="checkpoint_paths",
        type=str,
        nargs="*",
        help="checkpoint path",
    )
    checkpoint_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Disable the default update of the model entropy parameters before eval",
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    if args.source not in ["checkpoint", "pretrained"]:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    description = (
        "ELIC Model"
    )

    # filepaths = collect_images(args.dataset)
    # if len(filepaths) == 0:
    #     print("Error: no images found in directory.", file=sys.stderr)
    #     raise SystemExit(1)

    # compressai.set_entropy_coder(args.entropy_coder)

    criterion = RateDistortionLoss()
    device = 'cuda' if(args.cuda and torch.cuda.is_available()) else 'cpu'
    
    
    test_dataset = ImageFolder(args.dataset, split="test", transform=transforms.ToTensor())
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=1,
        shuffle=False,
        pin_memory=(device == "cuda"),
        collate_fn= custom_collate,
    )
    print(f'Number of images: {len(test_dataset)}')
    
    # create output directory
    if args.output_directory:
        Path(args.output_directory).mkdir(parents=True, exist_ok=True)


    model = ELICModel()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.update()
    model.eval()
    model.to(device)

    

    args_dict = vars(args)
    results, metrics_list = eval_model(
        model,
        args.output_directory,
        test_dataloader,
        criterion,
        device,
        **args_dict,
    )
    

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "ELIC_MODEL"
    )
    output = {
        "name": f"ELIC_MODEL-{args.metric}",
        "description": f"Inference ({description})",
        "results": metrics_list,
    }
    if args.output_directory:
        output_file = (
            args.output_file
            if args.output_file
            else f"{description}"
        )

        with (Path(f"{args.output_directory}/{output_file}").with_suffix(".json")).open(
            "wb"
        ) as f:
            f.write(json.dumps(output, indent=2).encode())

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])