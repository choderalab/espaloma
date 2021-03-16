for data in bayer benchmark zinc emolecules fda pfizer roche coverage
do
    echo $data
    mkdir $data
    let n_molecules=$(ls "_"$data | wc -l)
    echo $n_molecules
    bsub -J $data"[1-"$n_molecules"]" -o %J.stdout -W 2:00 python transform.py "_"$data/\$LSB_JOBINDEX $data/\$LSB_JOBINDEX
done
