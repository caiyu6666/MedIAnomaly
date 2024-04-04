num_repeat=1
datasets="rsna vin brain lag isic c16 brats"
#datasets="brats rsna"
settings="CutPaste FPI FPI-Poisson Shift-Intensity-M"
#settings="Shift-Intensity-M FPI-Poisson"
gpu_id=5

for data in $datasets;do
  for setting in $settings;do
    for((i=0;i<$num_repeat;i=i+1));do
    python train_med.py -d $data -s $setting -g $gpu_id -f $i;
    python med_evaluation.py -d $data -s $setting -g $gpu_id -f $i;
    done
  done
done
