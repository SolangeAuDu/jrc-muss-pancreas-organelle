bsub -J "flyrun03" -P cellmap -n 12 -q gpu_h100 -gpu "num=1" -o output.log -e error.log python train.py
