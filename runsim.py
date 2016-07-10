#!/usr/bin/env python
import radiusfn as rf
import pandas as pd
import numpy as np

import sys,os,os.path

outfilename = 'npps_sims2/results_%s.h5' % sys.argv[1]

if os.path.exists(outfilename):
    exit

df = rf.sim_ensemble(N=10)

df.to_hdf(outfilename,'table')

