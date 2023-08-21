# shellcheck disable=SC1009
# shellcheck disable=SC1061
# shellcheck disable=SC1073

num_repeat=3

# shellcheck disable=SC2004
for((i=0;i<$num_repeat;i=i+1));do
python train.py -m constrained-ae --input-size 64 -ls 16 --en-depth 1 --de-depth 1 -d brain -g 6 -f $i;
python train.py -m ceae --input-size 64 -ls 16 --en-depth 1 --de-depth 1 -d brain -g 6 -f $i;
done
