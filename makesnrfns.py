#!/usr/bin/env python

import radiusfn as rf
import occurrence as occ
import os,sys

i,n = (int(sys.argv[1]),int(sys.argv[2]))

allkois = rf.ALLKOIS
kois = allkois[i-1::n]
pdist = occ.ToyLogPdist(a=0.5,maxp=rf.MAXP)
#pdist = rf.logper_kde()
#rf.makeall_snrfns(kois,folder='koi_snrfns_q1q8_pfixed',Pfixed=True,pdist=pdist,maxq=8)
rf.makeall_snrfns(kois,folder='koi_snrfns_q1q8_ptoy',Pfixed=False,pdist=pdist,maxq=8)
#rf.makeall_snrfns(kois,folder='koi_snrfns_q1q8',Pfixed=False,pdist=pdist,maxq=8,overwrite=True)
