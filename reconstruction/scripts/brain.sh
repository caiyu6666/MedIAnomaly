# shellcheck disable=SC2034
# shellcheck disable=SC1073
# shellcheck disable=SC1009
# shellcheck disable=SC2004
data="brain"
input_size=64
gpu_id=5
num_repeat=3

for((i=1;i<=$num_repeat;i=i+1));do
  # standard baseline
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 16 --en-depth 1 --de-depth 1 -d $data -g $gpu_id;
  # latent size
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 1 --en-depth 1 --de-depth 1 -d $data -g $gpu_id;
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 2 --en-depth 1 --de-depth 1 -d $data -g $gpu_id;
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 4 --en-depth 1 --de-depth 1 -d $data -g $gpu_id;
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 8 --en-depth 1 --de-depth 1 -d $data -g $gpu_id;
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 32 --en-depth 1 --de-depth 1 -d $data -g $gpu_id;
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 64 --en-depth 1 --de-depth 1 -d $data -g $gpu_id;
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 128 --en-depth 1 --de-depth 1 -d $data -g $gpu_id;
  # depth
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 16 --en-depth 2 --de-depth 2 -d $data -g $gpu_id;
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 16 --en-depth 3 --de-depth 3 -d $data -g $gpu_id;
  # expansion
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 16 --en-depth 1 --de-depth 1 -d $data -g $gpu_id --expansion 2;
  python train.py -n ae-architecture -m ae --input-size $input_size -ls 16 --en-depth 1 --de-depth 1 -d $data -g $gpu_id --expansion 4;
done
