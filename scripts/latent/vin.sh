# shellcheck disable=SC2034
# shellcheck disable=SC1073
# shellcheck disable=SC1009
# shellcheck disable=SC2004
data="vin"
input_size=64
gpu_id=4
num_repeat=3
#ls=(1 2 4 8 16 32 64 128 256 512 1024)
#f=(0 1 2 3 4 5 6 7 8 9 10)
ls=(1 2 4 8 16 32 64)
f=(0 1 2 3 4 5 6)

for((j=0;j<$num_repeat;j=j+1));do
  for i in "${f[@]}";do
    python train.py -m ae -ls ${ls[$i]} -d $data -g $gpu_id -f "$i-$j";
  done
done