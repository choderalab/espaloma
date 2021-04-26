for name in malonaldehyde # benzene toluene malonaldehyde salicylic aspirin ethanol uracil
do
    for first in 50000 # 1 10 100 1000 10000 25000 50000 100000
    do
        for repeat in 0 # 1 2 3 4
        do
        bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R V100 -R "rusage[mem=10] span[ptile=1]" -W 12:00 -n 1\
        python run.py --name $name --first $first --n_epoch 3000 --out "_"$name"_"$first"_"$repeat
    done
    done
done
