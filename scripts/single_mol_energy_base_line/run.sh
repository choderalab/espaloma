#BSUB -q cpuqueue
#BSUB -o %J.stdout

bsub -q gpuqueue -m "ld-gpu ls-gpu lt-gpu lg-gpu lu-gpu" -n 4 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=4] span[hosts=1]" -W 0:30 -o %J.stdout -eo %J.stderr python train.py --n_epochs 100


