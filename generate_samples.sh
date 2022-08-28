#edit="big_eyes"

stylegan2_ckpt=checkpoint/stylegan2-ffhq-config-f.pt

truncation=0.8
sample=1
pics=20
alpha=0.25
seed=42
att_start=7
att_layer=18
att_channel=16

for edit in blond_hair brown_skin eyeliner eye_shadow green_eyes long_hair makeup pink_lipstick purple_hair red_beard white_teeth  
do
  for mask_thresh in 0 0.8
  do
    for train_iter in 01000 02000 05000 #10000 20000
    do
      python generate.py --ckpt $stylegan2_ckpt --truncation $truncation --sample $sample --pics $pics --alpha $alpha --seed $seed --mask_threshold $mask_thresh --train_iter $train_iter --clip_text $edit --att_layer $att_layer --att_start $att_start --att_channel $att_channel
    done
  done
done
