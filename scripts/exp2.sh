# shellcheck disable=SC1009
# shellcheck disable=SC1061
# shellcheck disable=SC1073

num_repeat=3
#data='rsna'
gpu=5

# shellcheck disable=SC2004
for((i=0;i<$num_repeat;i=i+1));do
#python train.py -m ae-perceptual -d $data -g $gpu -f $i;
#python train.py -d brain -g $gpu -m dae --input-size 128 -bs 16;
python train.py -d isic -g $gpu -m dae --input-size 128 -bs 16;
done
