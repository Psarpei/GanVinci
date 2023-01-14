# GanVinci

Reimplementation of the paper [FEAT: Face Editing with Attention](https://arxiv.org/abs/2202.02713) with additional changes and improvements.

## Setup
1. Clone this repository ```https://github.com/Psarpei/GanVinci.git```
2. CD into this repo: ```cd GanVinci```
3. Create conda environment from environment.yml ```conda env create -f environment.yml```
4. Download StyleGAN2 config-f weights from xxxxx /home/pascal/code/GanVinci/stylegan2-ffhq-config-f.pt 
5. Place weights under ```checkpoints/```

## Training
To train an text guided image editing (e.g. ```beard```, ```smiling_person```, ```open_mouth```, ```blond hair``` execute:

    python3 train_FEAT.py

with the following parameters

* ```--clip_text``` type ```str```, help "edit text e.g. beard, smile or open_mouth",
* ```--size``` output image size of the generator, type ```int```, default ```1024```
* ```--iterations``` number of samples to be generated for each image type ```int```, default ```20000```
* ```--truncation``` truncation ratio type ```float```, default ```1``` 
* ```--truncation_mean``` number of vectors to calculate mean for the truncation, type ```int```, default ```4096```
* ```--stylegan2_ckpt``` path to the StyleGAN2 model checkpoint, type ```str```, default ```stylegan2-ffhq-config-f.pt```
* ```--channel_multiplier``` channel multiplier of the generator. config-f = 2, else = 1, type ```int```, default ```2```
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
* ```--male_only``` flag that only uses images of male people
* ```--female_only``` flag that only uses images of female people
* ```--clip_only_steps``` amount of steps training only using clip loss for better convergence in some edits, type ```int```, default ```0```

## Usage

First create lmdb datasets:

> python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH

This will convert images to jpeg and pre-resizes it. This implementation does not use progressive growing, but you can create multiple resolution datasets using size arguments with comma separated lists, for the cases that you want to try another resolutions later.

Then you can train model in distributed settings

> python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH

train.py supports Weights & Biases logging. If you want to use it, add --wandb arguments to the script.

#### SWAGAN

This implementation experimentally supports SWAGAN: A Style-based Wavelet-driven Generative Model (https://arxiv.org/abs/2102.06108). You can train SWAGAN by using

> python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --arch swagan --batch BATCH_SIZE LMDB_PATH

As noted in the paper, SWAGAN trains much faster. (About ~2x at 256px.)

### Convert weight from official checkpoints

You need to clone official repositories, (https://github.com/NVlabs/stylegan2) as it is requires for load official checkpoints.

For example, if you cloned repositories in ~/stylegan2 and downloaded stylegan2-ffhq-config-f.pkl, You can convert it like this:

> python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl

This will create converted stylegan2-ffhq-config-f.pt file.

### Generate samples

> python generate.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT

You should change your size (--size 256 for example) if you train with another dimension.

### Project images to latent spaces

> python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...

### Closed-Form Factorization (https://arxiv.org/abs/2007.06600)

You can use `closed_form_factorization.py` and `apply_factor.py` to discover meaningful latent semantic factor or directions in unsupervised manner.

First, you need to extract eigenvectors of weight matrices using `closed_form_factorization.py`

> python closed_form_factorization.py [CHECKPOINT]

This will create factor file that contains eigenvectors. (Default: factor.pt) And you can use `apply_factor.py` to test the meaning of extracted directions

> python apply_factor.py -i [INDEX_OF_EIGENVECTOR] -d [DEGREE_OF_MOVE] -n [NUMBER_OF_SAMPLES] --ckpt [CHECKPOINT] [FACTOR_FILE]

For example,

> python apply_factor.py -i 19 -d 5 -n 10 --ckpt [CHECKPOINT] factor.pt

Will generate 10 random samples, and samples generated from latents that moved along 19th eigenvector with size/degree +-5.

![Sample of closed form factorization](factor_index-13_degree-5.0.png)

## Pretrained Checkpoints

[Link](https://drive.google.com/open?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO)

I have trained the 256px model on FFHQ 550k iterations. I got FID about 4.5. Maybe data preprocessing, resolution, training loop could made this difference, but currently I don't know the exact reason of FID differences.

## Samples

![Sample with truncation](doc/sample.png)

Sample from FFHQ. At 110,000 iterations. (trained on 3.52M images)

![MetFaces sample with non-leaking augmentations](doc/sample-metfaces.png)

Sample from MetFaces with Non-leaking augmentations. At 150,000 iterations. (trained on 4.8M images)

### Samples from converted weights

![Sample from FFHQ](doc/stylegan2-ffhq-config-f.png)

Sample from FFHQ (1024px)

![Sample from LSUN Church](doc/stylegan2-church-config-f.png)

Sample from LSUN Church (256px)

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
