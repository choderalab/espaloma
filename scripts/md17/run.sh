for name in benzene
do
    for first in 1 10 100 1000 10000 25000 50000 100000
    do
        bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 8:00 -n 1\
        python run.py --name $name --first $first
    done
done
