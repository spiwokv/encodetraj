import pytest
import mdtraj as md
import numpy as np
import keras as krs
import argparse as arg
import datetime as dt
import sys
import os

import encodetrajlib

def test_100_structures():
  myinfilename = os.path.join(os.path.dirname(__file__), 'traj_fit_small.xtc')
  myintopname = os.path.join(os.path.dirname(__file__), 'reference.pdb')
  ae, cor = encodetrajlib.encodetrajectory(infilename=myinfilename,
                                           intopname=myintopname,
                                           boxx=1, boxy=1, boxz=1, epochs=2000)
  assert(cor > 0.95)

if __name__ == '__main__':
  pytest.main([__file__])



