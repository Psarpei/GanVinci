import os
import argparse

import torch
from torchvision import utils
from model import FEAT, Generator
from torch import nn, autograd, optim

import clip
from torch.nn import CosineSimilarity
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)

class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        cos_distance = 1 - self.model(image, text)[0] / 100
        return cos_distance.mean()


def clip_loss(image, text_tokens, clip_model, cos_sim):
    
    image_features = clip_model.encode_image(image)
    text_features = clip_model.encode_text(text_tokens)

    cos_dist = 1- cos_sim(image_features, text_features)
    
    return cos_dist.mean()

def clip_most_likely_cat(image, cat_tokens, clip_model):

    logits_per_image, _ = clip_model(image, cat_tokens)
    probs = logits_per_image.softmax(dim=-1)
    return probs[0]

def att_reg_loss(mask):
    att_loss = mask.squeeze(1).sum(axis=[-1,-2]) * (1/(mask.size(-1)*mask.size(-2)))

    return att_loss.mean() 


def latent_loss(latent, latent_feat, att_start, att_layer):

    l2_loss = torch.sqrt(torch.sum(
                            torch.square(latent[:,att_start:att_layer] - latent_feat[:,att_start:att_layer]),
                             dim=-1
                            )
                        ).sum(dim=-1)

    return l2_loss.mean()


def total_variation_loss(mask):

    mask_squee = mask.squeeze(1)

    grad1 = (mask_squee[..., 1:, :] - mask_squee[..., :-1, :]).abs().sum(dim=[-1,-2])
    grad2 = (mask_squee[..., :, 1:] - mask_squee[..., :, :-1]).abs().sum(dim=[-1,-2])

    tv_loss = grad1 + grad2 

    return tv_loss.mean()


def train(args, feat, device, mean_latent, optimizer, scheduler, clip_model, text_tokens, cos_sim, clip_loss_s, mse, mf_tokens):
    log_string = "iteration: {:5}, loss: {:2.6} l_clip: {:2.6f}, l_att: {:2.6f}, l_l2: {:2.6f}, l_tv: {:2.6f}"

    clip_losses = []
    att_losses = []
    l2_losses = []
    tv_losses = []
    losses = []

    i = 1
    while i <= args.iterations:
        #reset gradients
        optimizer.zero_grad()
        
        sample_z = torch.randn(args.batch_size, args.latent, device=device)

        sample, edit_sample, mask, latent, latent_feat = feat(
            [sample_z],
            truncation=args.truncation,
            truncation_latent=mean_latent,
            return_latents=True,
            alpha=args.alpha
        )

        sample_resized = F.interpolate(sample,
                                     size=(224, 224),
                                     mode='bilinear')

        probs = clip_most_likely_cat(sample_resized, mf_tokens, clip_model)
        
        if(args.male_only and probs[0] < 0.8):
            continue
        
        #l_clip = clip_loss(edit_resized, text_tokens, clip_model, cos_sim)
        l_clip = clip_loss_s(edit_sample, text_tokens)

        l_att = att_reg_loss(mask)
        #l_l2 = latent_loss(latent, latent_feat, args.att_start, args.att_layer)
        l_l2 = mse(latent[:,args.att_start:args.att_layer], latent_feat[:,args.att_start:args.att_layer])
        l_tv = total_variation_loss(mask)


        """
        if i < 300:
            loss = l_clip
        else:
            loss = l_clip + args.lambda_l2 * l_l2 + args.lambda_att * l_att + args.lambda_tv * l_tv 
        """

        loss = l_clip + args.lambda_l2 * l_l2 + args.lambda_att * l_att + args.lambda_tv * l_tv 

        #print('old')
        #print(log_string.format(i, loss, l_clip, args.lambda_att * l_att, args.lambda_l2 * l_l2, args.lambda_tv * l_tv))
        #print(log_string.format(i, loss, l_clip, l_att, l_l2, l_tv))
        #print('new')
        print(log_string.format(i, loss, l_clip, args.lambda_att * l_att, args.lambda_l2 * l_l2, args.lambda_tv * l_tv))
        print(log_string.format(i, loss, l_clip, l_att, l_l2, l_tv))
        print()

        #backpropagation
        loss.backward()
    
        #update the parameters
        optimizer.step()
        scheduler.step()

        clip_losses.append(l_clip.item())
        att_losses.append(l_att.item())
        l2_losses.append(l_l2.item())
        tv_losses.append(l_tv.item())
        losses.append(loss.item())

        if(i in [1000, 2000, 5000, 10000, 20000]):
            torch.save(
                        {
                            "weights": feat.state_dict(),
                            "optim": optimizer.state_dict(),
                            "args": args,
                            "l_clip": clip_losses,
                            "l_att": att_losses,
                            "l_l2": l2_losses,
                            "l_tv": tv_losses,
                            "loss": losses
                        },
                        f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/checkpoints/{str(i).zfill(5)}_{args.clip_text}.pt",
                    )
        i+=1


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20000,
        help="number of samples to be generated for each image",
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--stylegan2_ckpt",
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
        "--FEAT_out_ckpt",
        type=str,
        default="FEAT"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001
    )
    parser.add_argument(
        "--lambda_att",
        type=float,
        default=0.005
    )
    parser.add_argument(
        "--lambda_tv",
        type=float,
        default=0.00001
    )
    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--clip_text",
        type=str
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
        "--lr_step_size",
        type=int,
        default=5000
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--male_only",
        action="store_true"
    )  
    parser.set_defaults(male_only=False)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    assert not args.male_only or args.batch_size == 1, "male_only is only possible at batch_size 1"

    #initialize generator
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.stylegan2_ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])
    
    #initialize FEAT
    feat = FEAT(g_ema, att_start=args.att_start, att_layer=args.att_layer, att_channel=args.att_channel).to(device)

    #initialize clip model prerequisites
    clip_model, _ = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize([args.clip_text]).to(device)
    mf_tokens = clip.tokenize(['male', 'female']).to(device)
    cos_sim = CosineSimilarity()

    #calculate mean if using truncation
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    #freeze generator
    for param in feat.generator.parameters():
        param.requires_grad = False

    #initialize optimizer
    optimizer = optim.Adam(
        feat.parameters(),
        lr=args.lr,
    )

    #initialize lr-scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    clip_loss_s = CLIPLoss()

    mse =  nn.MSELoss()

    #create save directories
    if not os.path.exists(f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/checkpoints"):
        os.makedirs(f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/checkpoints/")

    if not os.path.exists(f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/samples"):
        os.makedirs(f"edits/{args.att_start}-{args.att_layer}/{args.clip_text}/samples")

    train(args, feat, device, mean_latent, optimizer, scheduler, clip_model, text_tokens, cos_sim, clip_loss_s, mse, mf_tokens)