num_repeat=1
datasets="rsna vin brain lag isic c16 brats"
methods="ae ae-l1 ae-ssim ae-perceptual ae-spatial vae constrained-ae memae ceae ganomaly aeu ae-grad vae-rec vae-combi"
gpu=7

for data in $datasets;do
  for method in $methods;do
    for((i=0;i<num_repeat;i=i+1));do
      python train.py -d "$data" -m "$method" -g $gpu -f "$i";
      python test.py -d "$data" -m "$method" -g $gpu -f "$i" -save;
    done
  done

  for((i=0;i<num_repeat;i=i+1));do
      python train.py -d "$data" -m dae -g $gpu --input-size 128 -bs 16 -f "$i";
      python test.py -d "$data" -m dae -g $gpu --input-size 128 -f "$i" -save;
  done
done
