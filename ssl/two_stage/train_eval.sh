datasets="rsna vin brain lag isic c16 brats"
#datasets="brats rsna"
gpu_id=5
methods="normal 3way anatpaste"

for data in $datasets;do
  for method in $methods;do
    python run_training.py --variant $method --type $data --no-pretrained --cuda $gpu_id;
  done
done