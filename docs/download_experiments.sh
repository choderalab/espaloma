export fileid=1qdHEypk3uMhZEYCStWTU8u1uIDHzH3Qy
wget -O typing.ipynb 'https://docs.google.com/uc?export=download&id='$fileid
ipython nbconvert typing.ipynb --to rst --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}'
mv typing.rst experiments/typing.rst

export fileid=1krhwGHKoqL5-_P0G89fDB7Iw3ENHW2G_
wget -O mm_fitting_small.ipynb 'https://docs.google.com/uc?export=download&id='$fileid
ipython nbconvert mm_fitting_small.ipynb --to rst --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}'
mv mm_fitting_small.rst experiments/mm_fitting_small.rst
mv mm_fitting_small_files experiments/mm_fitting_small_files

export fileid=1i_z0b0-m_91bMww1hY5Kdc76VHmtHsWD
wget -O qm_fitting.ipynb 'https://docs.google.com/uc?export=download&id='$fileid
ipython nbconvert qm_fitting.ipynb --to rst --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}'
cp qm_fitting.rst experiments/qm_fitting.rst

rm *.ipynb
