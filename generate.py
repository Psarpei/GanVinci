import argparse

import torch
from torchvision import utils
from model import FEAT, Generator
from tqdm import tqdm


def generate(args, feat, device, mean_latent):

    with torch.no_grad():
        feat.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, edited_sample, mask, _ ,_ = feat(
                [sample_z],
                truncation=args.truncation,
                truncation_latent=mean_latent,
                alpha=args.alpha,
                mask_threshold=args.mask_threshold
            )

            utils.save_image(
                sample,
                f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/samples/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                edited_sample,
                f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/samples/feat_train_iter{args.train_iter}_mask_thresh{round(args.mask_threshold,2)}_{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                mask,
                f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/samples/mask_train_iter{args.train_iter}_mask_thresh{round(args.mask_threshold,2)}_{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            print('mask')
            print(mask)
            print(mask.max())


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--att_layer",
        type=int,
        default=8
    )
    parser.add_argument(
        "--att_channel",
        type=int,
        default=32
    )
    parser.add_argument(
        "--att_start",
        type=int,
        default=0
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--clip_text",
        type=str,
        default=""
    )
    parser.add_argument(
        "--train_iter",
        type=str,
        default=""
    )    

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    if args.seed:
        torch.manual_seed(args.seed)

    print("size", args.size, "latent", args.latent, "n_mlp", args.n_mlp, "channel_multiplier", args.channel_multiplier)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    feat = FEAT(g_ema, att_start=args.att_start, att_layer=args.att_layer, att_channel=args.att_channel).to(device)
    
    if(args.clip_text):
        feat_checkpoint=torch.load(f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/checkpoints/{args.train_iter}_{args.clip_text}.pt")
        feat.load_state_dict(feat_checkpoint["weights"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, feat, device, mean_latent)
