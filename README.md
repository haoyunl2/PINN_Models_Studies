# PINN_Models_Studies

# Set up the environments

Request an interactive computational node:
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=mxxxx_g
```
Load the modules:

```
module load cudatoolkit/21.9_11.4
module load python
module load pytorch
pip install yaml
```

# Study from PINO
Here I try to use Cosine Layer, which adapts Discrete Cosine Transformation, to compare the performance with PINO, which utilizes Fourier Layer.


