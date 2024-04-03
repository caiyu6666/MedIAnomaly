# shellcheck disable=SC1009
# shellcheck disable=SC1061
# shellcheck disable=SC1073

num_repeat=3
data='isic'
gpu=7

# shellcheck disable=SC2004
for((i=0;i<$num_repeat;i=i+1));do
python train.py -m ae-perceptual -d $data -g $gpu -f $i;
#python train.py -m ae-l1 -d $data -g $gpu -f $i;
#python train.py -m ae-ssim -d $data -g $gpu -f $i;
#python train.py -m vae -d $data -g $gpu -f $i;
#python train.py -m constrained-ae -d $data -g $gpu -f $i;
#python train.py -m ae-grad -d $data -g $gpu -f $i;
#python train.py -m vae-rec -d $data -g $gpu -f $i;
#python train.py -m vae-combi -d $data -g $gpu -f $i;
#python train.py -m ceae -d $data -g $gpu -f $i;
#python train.py -m ganomaly -d $data -g $gpu -f $i;
#python train.py -m memae -d $data -g $gpu -f $i;
#python train.py -m aeu -d $data -g $gpu -f $i;
#python train.py -m ae-spatial -d $data -ls 1 --hidden-num 64 -g $gpu -f $i;
#python train.py -m ae-spatial -d $data -ls 2 --hidden-num 64 -g $gpu -f $i;
done
