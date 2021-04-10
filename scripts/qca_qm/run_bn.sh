#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=30] span[ptile=1]"
#BSUB -W 8:00
#BSUB -n 1

n_epochs=3000
layer="SAGEConv"
units=128
act="relu"
lr=1e-4
graph_act="relu"
weight=1.0
first=10

python run.py --n_epochs $n_epochs --layer $layer --config $units "bn" $act $units "bn" $act $units "bn" $act --janossy_config $units "bn" $act $units "bn" $act $units "bn" $act --weight $weight --out $units"_"$layer"_"$act"_"$weight"_"$lr"_"$n_epochs"_"$small_batch"_"$big_batch"_single_gpu_janossy_first_distributed"$first"bn" --lr $lr --graph_act $act --first $first



