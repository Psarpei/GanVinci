# GanVinci
Photorealistic attention based text guided human image editing with a latent mapper for StyleGAN2.

![image of my project](https://github.com/Psarpei/GanVinci/blob/master/GanVinci2.png)

This work is a reimplementation of the paper [FEAT: Face Editing with Attention](https://arxiv.org/abs/2202.02713) with additional changes and improvements.

## Setup
1. Clone this repository ```https://github.com/Psarpei/GanVinci.git```
2. CD into this repo: ```cd GanVinci```
3. Create conda environment from environment.yml ```conda env create -f environment.yml```
4. Download StyleGAN2 config-f weights from [here](https://drive.google.com/drive/folders/1C58moKK0AOri27DuhZPEKxHC8ea9SRAF?usp=share_link)
5. Place StyleGAN2 weights under ```checkpoints/```

## Training
To train an text guided image editing (e.g. ```beard```, ```smiling_person```, ```open_mouth```, ```blond_hair``` execute:

    python3 train_FEAT.py

with the following parameters

* ```--clip_text``` type ```str```, help "edit text e.g. beard, smile or open_mouth",
* ```--batch_size```batch size (need to be one if --(fe)male_only is activated, type ```int```, default ```1``` 
* ```--lr``` learnrate, type ```float```, default=0.0001
* ```--lambda_att``` latent attention regression loss factor, type ```float```, default=0.005
* ```--lambda_tv``` total variation loss factor, type ```float```, default ```0.00001```
* ```--lambda_l2``` l2 loss factor, type ```float```, default ```0.8```
* ```--att_layer``` layer of attention map, type ```int```, default ```8```
* ```--att_channel``` number of channels of attention map, type ```int```, default ```32```
* ```--att_start``` start attention layer of the latent mapper, type ```int``` default ```0```
* ```--lr_step_size``` learning rate step size for scheduler, type ```int```, default ```5000```
* ```--lr_gamma``` gamma for learning rate of scheduler, type ```float```, default ```0.5```
* ```--alpha``` factor of latent mapper type ```float```, default ```0.5```
* ```--clip_only_steps``` amount of steps training only using clip loss for better convergence in some edits, type ```int```, default ```0```
* ```--size``` output image size of the generator, type ```int```, default ```1024```
* ```--iterations``` number of samples to be generated for each image, type ```int```, default ```20000```
* ```--truncation``` truncation ratio, type ```float```, default ```1``` 
* ```--truncation_mean``` number of vectors to calculate mean for the truncation, type ```int```, default ```4096```
* ```--stylegan2_ckpt``` path to the StyleGAN2 model checkpoint, type ```str```, default ```stylegan2-ffhq-config-f.pt```
* ```--channel_multiplier``` channel multiplier of the generator. config-f = 2, else = 1, type ```int```, default ```2```
* ```--male_only``` flag that only uses images of male people
* ```--female_only``` flag that only uses images of female people

In the ```bash_examples/```folder are a few inference invokes provided.

## Inference
For inference it is required to have the trained edit checkpoints placed under the folder structure like the following example

```
edits/
├── 0-8/
│   ├── beard/
│   │   │   ├──  checkpoints/  
│   │   │   │   ├── 01000_beard.pt
│   │   │   │   ├── 02000_beard.pt    
│   │   ... ... ...
│   │   │   │   └── 20000_beard.pt    
│   ...
...
```

To apply a trained text guided image edit execute:

    python3 generate.py

with the following parameters
* ```--clip_text``` name of edit (e.g. ```beard```, ```smile``` etc.), if "" standard styleGAN2 image generation is applied, type ```str```, default ```""```
* ```--alpha``` factor of latent mapper, type ```float```, default ```0.1``` 
* ```--att_layer``` layer of attention map, type ```int```, default ```8``` 
* ```--att_channel``` number of channels of attention map, type ```int```, default ```32```
* ```--att_start``` start attention layer of the latent mapper, type ```int```, default ```0```
* ```--mask_threshold``` threshold for mask apply based on predicted pixels,, type ```float```, default ```0.8```
* ```--train_iter```  iteration steps of edit checkpoint, type ```str```, default ```""```
* ```--size``` output image size of the generator, type ```int```, default ```1024```
* ```--sample``` number of samples to be generated for each image type, ```int```, default ```1``` 
* ```--pics``` number of images to be generated, type, ```int```, default ```20```
* ```--truncation``` truncation ratio, type ```float```, default ```1```
* ```--truncation_mean``` number of vectors to calculate mean for the truncation, type ```int```, default ```4096```
* ```--ckpt``` path to the model checkpoint, type ```str```, default ```stylegan2-ffhq-config-f.pt```
* ```--channel_multiplier``` channel multiplier of the generator. config-f = 2, else = 1, type ```int```, default ```2```
* ```--seed``` random seed for image generation, type ```int```, default ```0```
* ```--male_only``` flag that only uses images of female people
* ```--female_only``` flag that only uses images of female people

In the ```bash_examples/```folder are a few inference invokes provided.

## Pre-trained Edits
You can download some weights of pre-trained edits [here](https://drive.google.com/drive/folders/1O2cCwasxJ6H6vkgO4iAyZQOIwzEno9vD?usp=share_link).
To apply a pre-trained edit leave the folder structure as it is and place everything under ``edits/`` like explained in the inference section. 

## Acknowledgments
This code borrows heavily from [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) and the model is based on the paper [FEAT: Face Editing with Attention](https://arxiv.org/abs/2202.02713).
