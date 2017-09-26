#!/bin/sh


source activate aind-dog
KRAS_BACKEND=tensorflow python -c "from keras import backend"
jupyter notebook --ip=0.0.0.0 --no-browser&

