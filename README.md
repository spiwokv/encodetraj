## encodetraj
### (Deep learning) autoencoders for molecular trajectory analysis, requires numpy, keras and mdtraj

Example:
```
./encodetraj.py -i traj_fit.xtc -p reference.pdb -boxx 1 -boxy 1 -boxz 1 -testset 0.2 \ 
                -low low.txt -high high.txt -filter filtered -model model -epochs 1000
```
-> ![c8](https://github.com/spiwokv/encodetraj/raw/master/data/low_small.png) <-

Usage:
```
usage: encodetraj.py [-h] [-i INFILE] [-p INTOP] [-boxx BOXX] [-boxy BOXY]
                     [-boxz BOXZ] [-testset TESTSET] [-shuffle SHUFFLE]
                     [-layers LAYERS] [-layer1 LAYER1] [-layer2 LAYER2]
                     [-encdim ENCDIM] [-actfun1 ACTFUN1] [-actfun2 ACTFUN2]
                     [-optim OPTIM] [-loss LOSS] [-epochs EPOCHS]
                     [-batch BATCH] [-low LOWFILE] [-high HIGHFILE]
                     [-filter FILTERFILE] [-model MODELFILE]
                     [-plumed PLUMEDFILE]

(Deep learning) autoencoders for molecular trajectory analysis, requires
numpy, keras and mdtraj

optional arguments:
  -h, --help          show this help message and exit
  -i INFILE           Input trajectory in pdb, xtc, trr, dcd, netcdf or mdcrd,
                      WARNING: the trajectory must be 1. centered in the PBC
                      box, 2. fitted to a reference structure and 3. must
                      contain only atoms to be analysed!
  -p INTOP            Input topology in pdb, WARNING: the structure must be 1.
                      centered in the PBC box and 2. must contain only atoms
                      to be analysed!
  -boxx BOXX          Size of x coordinate of PBC box (from 0 to set value in
                      nm)
  -boxy BOXY          Size of y coordinate of PBC box (from 0 to set value in
                      nm)
  -boxz BOXZ          Size of z coordinate of PBC box (from 0 to set value in
                      nm)
  -testset TESTSET    Size of test set (fraction of the trajectory, default =
                      0.1)
  -shuffle SHUFFLE    Shuffle trajectory frames to obtain training and test
                      set (default True)
  -layers LAYERS      Number of encoding layers (same as number of decoding,
                      allowed values 2-3, default = 2)
  -layer1 LAYER1      Number of neurons in the second encoding layer (default
                      = 256)
  -layer2 LAYER2      Number of neurons in the third encoding layer (default =
                      256)
  -encdim ENCDIM      Encoding dimension (default = 3)
  -actfun1 ACTFUN1    Activation function of the first layer (default =
                      sigmoid, for options see keras documentation)
  -actfun2 ACTFUN2    Activation function of the second layer (default =
                      sigmoid, for options see keras documentation)
  -optim OPTIM        Optimizer (default = adam, for options see keras
                      documentation)
  -loss LOSS          Loss function (default = mean_squared_error, for options
                      see keras documentation)
  -epochs EPOCHS      Number of epochs (default = 100, >1000 may be necessary
                      for real life applications)
  -batch BATCH        Batch size (0 = no batches, default = 256)
  -low LOWFILE        Output file with low-dimensional embedings (xvg or txt,
                      default = no output)
  -high HIGHFILE      Output file with original coordinates and encoded-
                      decoded coordinates (xvg or txt, default = no output)
  -filter FILTERFILE  Output file with encoded-decoded trajectory in .xtc
                      format (default = no output)
  -model MODELFILE    Prefix for output model files (experimental, default =
                      no output)
  -plot PLOTFILE      Model plot file in png (default = no output)
  -plumed PLUMEDFILE  Output file for Plumed (default = no output) TODO
```

