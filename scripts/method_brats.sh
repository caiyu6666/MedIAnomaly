# shellcheck disable=SC1009
# shellcheck disable=SC1061
# shellcheck disable=SC1073

num_repeat=3
data='brats'
gpu=7

# shellcheck disable=SC2004
for((i=0;i<$num_repeat;i=i+1));do
python train.py -m ae-l1 -d $data -g $gpu -f $i;
done
