import argparse
from cgitb import text

import torch
from torchvision import utils
from model import FEAT, Generator
#from tqdm import tqdm

import clip
from torch.nn import functional as F


def clip_most_likely_cat(image, cat_tokens, clip_model):

    logits_per_image, _ = clip_model(image, cat_tokens)
    probs = logits_per_image.softmax(dim=-1)
    return probs[0]


def generate(args, feat, device, mean_latent, clip_model, mf_tokens): # text_tokens):

    with torch.no_grad():
        feat.eval()
        i = 0
        while i < args.pics:
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, edited_sample, mask, _ ,_ = feat(
                [sample_z],
                truncation=args.truncation,
                truncation_latent=mean_latent,
                alpha=args.alpha,
                mask_threshold=args.mask_threshold
            )

            if(args.male_only):
                sample_resized = F.interpolate(sample,
                                            size=(224, 224),
                                            mode='bilinear')

                probs = clip_most_likely_cat(sample_resized, mf_tokens, clip_model)

            if((args.male_only and probs[0] < 0.8) or (args.female_only and probs[1] < 0.8)):
                continue

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

            i+=1


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
        default=0.1,
        help="factor of latent mapper",
    )
    parser.add_argument(
        "--att_layer",
        type=int,
        default=8,
        help="layer of attention map",
    )
    parser.add_argument(
        "--att_channel",
        type=int,
        default=32,
        help="number of channels of attention map",
    )
    parser.add_argument(
        "--att_start",
        type=int,
        default=0,
        help="start attention layer of the latent mapper",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.8,
        help="threshold for mask apply based on predicted pixels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for image generation",
    )
    parser.add_argument(
        "--clip_text",
        type=str,
        default="",
        help="name of clip edit checkpoint, if 0 standard styleGAN2 image generation is applied"
    )
    parser.add_argument(
        "--train_iter",
        type=str,
        default="",
        help="iteration steps of edit checkpoint"
    )  
    parser.add_argument(
        "--male_only",
        action="store_true",
        help="flag that only uses images of female people"
    )  
    parser.add_argument(
        "--female_only",
        action="store_true",
        help="flag that only uses images of female people"
    ) 
    parser.set_defaults(male_only=False)
    parser.set_defaults(female_only=False)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    assert not args.male_only or args.sample == 1, "male_only is only possible at batch_size 1"

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

    mf_tokens = clip.tokenize(['male', 'female']).to(device) #only used if args.male_only

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    #initialize clip model prerequisites
    clip_model, _ = clip.load("ViT-B/32", device=device)
    #text_tokens = clip.tokenize(["male", "female"]).to(device)

    generate(args, feat, device, mean_latent, clip_model, mf_tokens)
