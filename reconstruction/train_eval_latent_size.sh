num_repeat=3
datasets="rsna vin brain lag isic c16 brats"
latent_sizes="1 2 4 8 16 32 64 128"
gpu=0


for data in $datasets;do
  for ls in $latent_sizes;do
    for((i=0;i<num_repeat;i=i+1));do
      python train.py -ls $ls -d "$data" -m ae -g $gpu -f "$i";
      python test.py -ls $ls -d "$data" -m ae -g $gpu -f "$i" -save;
    done
  done
done
