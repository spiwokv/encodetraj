import pytest
import mdtraj as md
import numpy as np
import keras as krs
import argparse as arg
import datetime as dt
import sys

import encodetrajlib

def test_100_structures():
  ae, cor = encodetrajlib.encodetrajectory(infilename='./traj_fit_small.xtc',
                                           intopname='./reference.pdb',
                                           boxx=1, boxy=1, boxz=1, epochs=2000)
  assert(cor > 0.95)

if __name__ == '__main__':
  pytest.main([__file__])



