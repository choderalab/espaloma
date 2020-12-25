#BSUB -q cpuqueue
#BSUB -o %J.stdout

for units in 128 # 256 512 1024
do
    for act in "sigmoid"  # "leaky_relu"  # 'sigmoid' # 'leaky_relu' 'tanh'
    do
        for layer in 'GraphConv' 'Free' # 'GraphConv' # 'SAGEConv' # 'GINConv' 'SGConv' 'EdgeConv'
        do
            for opt in "LBFGS"  # "Adam" "SGLD" # "LBFGS"
            do
                for metric in "param" "energy" # "force"
                do
                    for repeat in {0..4}
                    do

            bsub -q gpuqueue -n 1 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=6] span[hosts=1]" -W 48:00 -o %J.stdout -eo %J.stderr python run.py --layer $layer --config $units $act $units $act $units $act $units $act $units $act $units $act --metric $metric --opt $opt --out $layer"_"$opt"_"$metric"_"$repeat"_long" --n_epochs 10000 

done
done
done
done
done
done
