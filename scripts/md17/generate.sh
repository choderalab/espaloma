for name in benzene uracil naphthalene aspirin salicylic malonaldehyde ethanol\
    toluene paracetamol azobenzene
do
    bsub -W 8:00 -R "rusage[mem=10]" -o %J.stdout python run.py $name
done
