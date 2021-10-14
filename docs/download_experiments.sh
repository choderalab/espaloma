export fileid=1qdHEypk3uMhZEYCStWTU8u1uIDHzH3Qy
wget -O typing.ipynb 'https://docs.google.com/uc?export=download&id='$fileid
ipython nbconvert typing.ipynb --to rst
mv typing.rst experiments/typing.rst


rm *.ipynb
