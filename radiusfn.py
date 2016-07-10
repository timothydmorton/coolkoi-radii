#!/usr/bin/env python

import os,sys,re,os.path,shutil,glob
import pickle
FPPDIR = '%s/FPP' % os.environ['DROPBOX']
RESULTSDIR = '%s/results/coolkois.txt.roboao' % FPPDIR
#RESULTSDIR = '%s/results/coolkois.txt.nocc' % FPPDIR
if sys.path[0] != FPPDIR:
    sys.path.insert(0,'%s/src' % FPPDIR)

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print 'pylab not imported.'
try:
    import transitFPP as fpp
except:
    print 'transitFPP not imported...'
import parsetable as pt
import plotutils as plu
try:
    import keplerfpp as kfpp
except:
    print 'keplerfpp not imported...'
from consts import *
from scipy.interpolate import UnivariateSpline as interpolate
from scipy.stats import gaussian_kde
from scipy.integrate import quad,trapz
import numpy.random as rand
import utils
from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar

from scipy.optimize import leastsq

import matplotlib.ticker
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from astropy.table import Table,Column
import pandas as pd

import occurrence as occ
from consts import *

import koiutils_old as ku


JOVIANMASSRATS = dict(Io=0.00004705631,Europa=0.00002528804,
                      Ganymede=0.00007807157,Callisto=0.0000566821)
SSMASSRATS = dict(Earth=0.00000300245,Venus=0.00000244699,
                  Mercury=1.65956463e-7,Mars=3.22604696e-7)


DRESSINGPROPS = pt.parsetxt('%s/dressing_starprops.txt' % os.environ['FPPDIR'])
DRESSINGKOIS = pt.parsetxt('%s/dressing_kois.txt' % os.environ['FPPDIR'])
DRESSINGKOIS = ku.make_KOIdict(DRESSINGKOIS)

DRESSINGPROPS[8561063] = dict(FeH=-0.48,Mstar=0.13,Rstar=0.17,Teff=3068,Ep_Rstar=0.04,En_Rstar=0.04)  # KOI-961

DRESSINGPROPS_REC = np.recfromtxt('%s/dressing_starprops.txt' % os.environ['FPPDIR'],names=True)
DRESSINGKOIS_REC = np.recfromtxt('%s/dressing_kois.txt' % os.environ['FPPDIR'],names=True)


SWIFTREC = np.recfromtxt('long_fixlimb_fitparams.dat',names=True)
SWIFTPROPS = ku.rec2dict(SWIFTREC,'KOI')
USESWIFT = True

USE_Q1Q8 = False
try:
    f = open('kois_q1q8.pkl','rb')
    d = pickle.load(f)
    KOIS_Q1Q8 = ku.make_KOIdict(d)
#USE_Q1Q8 = False
except:
    print 'warning: KOIS_Q1Q8 not set up'

USE_Q1Q8 = False
USE_Q1Q12 = True
try:
    f = open('kois_q1q12.pkl','rb')
    d = pickle.load(f)
    KOIS_Q1Q12 = ku.make_KOIdict(d)
except:
    print 'warning: KOIS_Q1Q12 not set up'

if USE_Q1Q8:
    MAXQ = 8
    SNRRAMP = (6,12)
elif USE_Q1Q12:
    MAXQ = 12
    SNRRAMP = (6,10)
else:
    MAXQ = 6
    SNRRAMP = (6,16)

ALLKOIS = DRESSINGKOIS.keys() + ['KOI961.01','KOI961.02','KOI961.03']
#ALLKOIS.remove('KOI1686.01')

#MUIRHEADKOIS = pt.parse_coolkois()  #make MUIRHEADKOIS a KOIdict

t = fits.getdata('cool_koi_parameters_new.fits')
MUIRHEADKOIS = ku.KOIdict()
for i,k in enumerate(t['KEPOI_NAME']):
    d = {}
    d['period'] = t['KOI_PERIOD'][i]
    d['Teff'] = t['TEFF_ROJAS'][i]
    d['e_Teff_plus'] = t['E_TEFF_ROJAS'][i]
    d['e_Teff_minus'] = t['E_TEFF_ROJAS'][i]
    d['M'] = t['MSTAR'][i]
    d['e_M'] = t['E_MSTAR'][i]
    d['R'] = t['RSTAR'][i]
    d['e_R'] = t['E_RSTAR'][i]
    d['feh'] = t['FEH_ROJAS'][i]
    d['e_feh'] = t['E_FEH_ROJAS'][i]
    d['duration'] = t['KOI_DURATION'][i]
    d['Rp'] = t['RPL'][i]
    d['e_Rp'] = t['E_RPL'][i]
    d['kic'] = t['KEPID'][i]
    d['a'] = t['A'][i]
    d['arstar'] = d['a']*AU/(d['R']*RSUN)
    MUIRHEADKOIS[k] = d
    

NOPHIL = False

BADKOIS = ['KOI886.01','KOI886.02','KOI961.01','KOI961.02','KOI961.03']

#don't use phil's spectra above 3800 K!

try:
    if USE_Q1Q8:
        ALLKOIS = KOIS_Q1Q8.keys()
    elif USE_Q1Q12:
        ALLKOIS = KOIS_Q1Q12.keys()
except:
    print 'warning: ALLKOIS not set up.'

#import koiutils as ku
#KOIS_Q1Q12 = ku.csv2dict('kois_q1q12.csv','kepoi_name')

#KOIS_Q1Q12 = 

#if USE_Q1Q12:
#    ALLKOIS = 

KICS = np.zeros(len(KOIS_Q1Q12))
RPS = np.zeros(len(KOIS_Q1Q12))
DRPS_P = np.zeros(len(KOIS_Q1Q12))
DRPS_N = np.zeros(len(KOIS_Q1Q12))
PERS = np.zeros(len(KOIS_Q1Q12))
for i,k in enumerate(KOIS_Q1Q12.keys()):
    KICS[i] = KOIS_Q1Q12[k]['kic']
    DRPS_P[i] = KOIS_Q1Q12[k]['Ep_rp']
    DRPS_N[i] = KOIS_Q1Q12[k]['En_rp']
    RPS[i] = KOIS_Q1Q12[k]['rp']
    PERS[i] = KOIS_Q1Q12[k]['P']

KOIS_Q1Q12_TABLE = Table(data=[KOIS_Q1Q12.keys(),KICS,RPS,DRPS_P,DRPS_N,PERS],
                         names=['name','ID','Rp','dRp_p','dRp_n','P'])

SWIFT_STARDATA = pd.read_table('coolkoi_starpars.txt',sep='\s+',index_col=0)

MAXP = 150


class CoolKOISurvey(occ.TransitSurveyFromASCII,occ.TransitSurvey):
    def __init__(self,maxq=MAXQ,recalc=False,etadisc_filename='coolkois_q1q12_etadisc.txt',
                 tag=None,snrdist_filename='coolkois_q1q12_snrdist.fits',logperdist=None,
                 maxp=MAXP,
                 **kwargs):
        if tag is not None:
            etadisc_filename = 'coolkois_q1q12_etadisc_%s.txt' % tag
            snrdist_filename = 'coolkois_q1q12_snrdist_%s.fits' % tag

        if os.path.exists('coolkois_q1q12_targets.txt') and \
           os.path.exists('coolkois_q1q12_detections.txt') and \
           not recalc:
            occ.TransitSurveyFromASCII.__init__(self,'coolkois_q1q12_targets.txt',
                                                'coolkois_q1q12_detections.txt',
                                                etadisc_filename=etadisc_filename,
                                                snrdist_filename=snrdist_filename,
                                                logperdist=logperdist,maxp=maxp,
                                                survey_transitprob=0.026,**kwargs)
        else:
            kics = DRESSINGPROPS_REC['KID']
            Rs = DRESSINGPROPS_REC['Rstar']
            Ms = DRESSINGPROPS_REC['Mstar']
            Teffs = DRESSINGPROPS_REC['Teff']
            loggs = DRESSINGPROPS_REC['logg']
            fehs = DRESSINGPROPS_REC['FeH']
            Tobs = np.zeros(len(kics))
            noise = np.zeros(len(kics))
            for i,k in enumerate(kics):
                Tobs[i] = kfpp.days_observed(k,maxq=maxq)
                noise[i] = kfpp.median_CDPP(k,maxq=maxq)

            targets = Table(data=[kics,Rs,Ms,Teffs,loggs,fehs,Tobs,noise],
                            names=['ID','R','M','Teff','logg','feh','Tobs','noise'])
            targets = targets[np.where(targets['Tobs']>0)]

            detections = KOIS_Q1Q12_TABLE


            occ.TransitSurvey.__init__(self,targets,detections,etadisc_filename=etadisc_filename,
                                       survey_transitprob=0.026,recalc_etadisc=recalc,
                                       snrdist_filename=snrdist_filename,
                                       logperdist=logperdist,maxp=maxp,
                                       recalc=recalc,**kwargs)

            self.writetargets_ascii()
            self.writedetections_ascii()

    def writetargets_ascii(self,filename='coolkois_q1q12_targets.txt'):
        occ.TransitSurvey.writetargets_ascii(self,filename)
        
    def writedetections_ascii(self,filename='coolkois_q1q12_detections.txt'):
        occ.TransitSurvey.writedetections_ascii(self,filename)
        

class CoolKOIRadiusFunction(occ.RadiusKDE_FromSurvey):
    def __init__(self,maxq=MAXQ,recalc_survey=False,widthfactor=1,rmin=0.,rmax=8,
                 tag=None,logperdist=None,maxp=MAXP,
                 **kwargs):
        s = CoolKOISurvey(maxq=maxq,recalc=recalc_survey,tag=tag,
                          logperdist=logperdist,maxp=maxp)

        print s.logperdist

        posteriors = []
        ##approximate kernels as 2-sided gaussians
        #for drn,drp in zip(s.detections['dRp_n'],s.detections['dRp_p']):
        #    kernels.append(occ.dists.DoubleGauss_Distribution(0,drn,drp))

        Pvals = []
        fps_initial = []
        fpp_calculated = []
        for k,rp,drn,drp in zip(s.detections['name'],s.detections['Rp'],
                                s.detections['dRp_n'],s.detections['dRp_p']):
            try:
                posteriors.append(koi_rp_posterior(k))
            except IOError:
                print 'no Rp PDF for %s; using "doublegauss"' % k
                posteriors.append(occ.dists.DoubleGauss_Distribution(rp,drn,drp))
            
            fp = ku.FPPDATA.ix[k,'fp_specific']
            if np.isnan(fp):
                fp = 0.2
                print 'using fp=0.2 as default for %s' % k
            fps_initial.append(fp)

            fpp_calculated.append(True)

            try:
                Pval = ku.Pval(k)
                if np.isnan(Pval):
                    err = ku.FPPDATA.ix[k,'error']
                    if err in ['MissingKOIError','DetectedCompanionError']:
                        Pval = (1-0.1)/(0.1*fp) #assuming default FPP=0.1
                    print 'Pval is nan for %s; provided error is: %s.  Assuming FPP=0.1' % \
                        (k,ku.FPPDATA.ix[k,'error'])
                    fpp_calculated[-1] = False
                #if FPP-assumed rp is way off, assign 10% FPP
                if np.absolute(np.log10(ku.FPPDATA.ix[k,'rp']/rp) > np.log10(1.3)):
                    Pval = (1-0.1)/(0.1*fp)
                    print 'FPP analysis for %s used inconsistent radius (not within 30 pct).  Using FPP=0.1' % k
                    fpp_calculated[-1] = False
                Pvals.append(Pval)
                
            except ku.NotOnTargetError:
                fpp = 1 - ((1-ku.FPPDATA.ix[k,'fpp']) * 
                           ku.FPPDATA.ix[k,'prob_ontarget']) #temporary hack-y, but OK solution...
                if np.isnan(fpp):
                    fpp = 1 - (0.9 * ku.FPPDATA.ix[k,'prob_ontarget']) 
                    print 'assuming fpp=0.1 for %s.' % k
                    fpp_caculated[-1] = False
                Pval = (1-fpp)/(fpp*fp)
                Pvals.append(Pval)
            except ku.FalsePositiveError:
                fpp_calculated[-1] = False
                Pvals.append(0)
            #print k,Pvals[-1]

        fps_initial = np.ones(len(Pvals))*.2
        self.fpp_calculated = np.array(fpp_calculated)
        occ.RadiusKDE_FromSurvey.__init__(self,s,widthfactor=widthfactor,minval=rmin,maxval=rmax,
                                          posteriors=np.array(posteriors),maxp=maxp,
                                          Pvals=np.array(Pvals),fps_initial=np.array(fps_initial),
                                          **kwargs)

    def write_tex(self,filename='rkde_table.tex',provenance=True):
        fout = open(filename,'w')
        inds = np.argsort(self.survey.detections['Rp'])

        #columns: KOI,Rp,dRp_p,dRp_m,etadisc,FPP,w
        for i in inds:
            koi = self.survey.detections['name'][i]
            rp = self.survey.detections['Rp'][i]
            drp_p = self.posteriors[i].pctile(0.84)-self.posteriors[i].pctile(0.5)
            drp_m = self.posteriors[i].pctile(0.5)-self.posteriors[i].pctile(0.16)
            eta = self.etadiscs[i]
            fpp = self.fpps[i]
            w = self.weights[i]
            if self.fpp_calculated[i]:
                fppstr = '%.2g' % fpp
            else:
                fppstr = '%.2g\\tablenotemark{c}' % fpp

            koinum = ku.koiname(koi,koinum=True)
            try:
                if SWIFT_STARDATA.ix[koinum,'Prov']=='P':
                    prov = 'a'
                elif SWIFT_STARDATA.ix[koinum,'Prov']=='C':
                    prov = 'b'
            except KeyError:
                print '%s not in swift stellar parameter file; using dressing parameters' % koi
                prov = 'b'
            fout.write('%.2f\\tablenotemark{%s} & %.2f & %.2f & %.2f & %.2f & %s & %.1f\\\\\n' % \
                       (koinum,prov,rp,drp_p,drp_m,eta,fppstr,w))

        fout.close()


def koi_rp_lorgauss(koi):
    rps,pdf = np.loadtxt('%s/%s_Rp_long.dat' % (ku.koiname(koi,koinum=True,star=True),ku.koiname(koi,koinum=True)),
                         unpack=True)
    pfit = occ.dists.fit_double_lorgauss(rps,pdf)
    return occ.dists.DoubleLorGauss_Distribution(*pfit,minval=rps.min(),maxval=rps.max(),name='Rp')

def koi_rp_posterior(koi):
    rps,pdf = np.loadtxt('%s/%s_Rp_long.dat' % (ku.koiname(koi,koinum=True,star=True),ku.koiname(koi,koinum=True)),
                         unpack=True)
    return occ.dists.EmpiricalDistribution(rps,pdf,name='Rp')
    
def jswift_table(fname='jswift_table.txt'):
    fout = open(fname,'w')
    fout.write('#koi rp period\n')
    for k in ALLKOIS:
        props = SNRprops(k)
        fout.write('%s %.2f %.3f\n' % (k,props['Rp'],props['period']))
    fout.close()

def MofR(r,comp='simple',folder='solidmrrelations'):
    if comp=='simple':
        return r**2.06
    elif comp=='earth':
        fname = '%s/outfecore32.5pmgsio3.dat' % folder
    elif comp=='fe':
        fname = '%s/outfe.dat' % folder
    elif comp=='h2o' or comp=='ice':
        fname = '%s/outh2o.dat' % folder
    elif comp=='rock':
        fname = '%s/outmgsio3.dat' % folder
    else:
        fname = '%s/%s.dat' % (folder,comp)
    ms,rs = np.loadtxt(fname,unpack=True,usecols=(0,1))
    r = np.atleast_1d(r)
    dr = rs[1:]-rs[:-1]
    #w = np.where(rs < r.max()*2)
    wbad = np.where(dr < 0)[0]
    imax = wbad[0]
    inds = range(imax+1)
    fn = interpolate(rs[inds],ms[inds],s=0)
    return fn(r)

try:
    ALLPERIODS = []
    ALLRPS = []
    ALLMASSRATIOS = []
    for k in ALLKOIS:
        if USE_Q1Q8:
            ALLPERIODS.append(KOIS_Q1Q8[k]['P'])
            ALLRPS.append(KOIS_Q1Q8[k]['rp'])
        elif USE_Q1Q12:
            ALLPERIODS.append(KOIS_Q1Q12[k]['P'])
            ALLRPS.append(KOIS_Q1Q12[k]['rp'])
        elif k not in DRESSINGKOIS or k in BADKOIS:
            ALLPERIODS.append(MUIRHEADKOIS[k]['period'])
            ALLRPS.append(MUIRHEADKOIS[k]['Rp'])
            ALLMASSRATIOS.append(MofR(MUIRHEADKOIS[k]['Rp'])*MEARTH/(MUIRHEADKOIS[k]['M']*MSUN))
        else:
            ALLPERIODS.append(DRESSINGKOIS[k]['P'])
            ALLRPS.append(DRESSINGKOIS[k]['rp'])
            ALLMASSRATIOS.append(MofR(DRESSINGKOIS[k]['rp'])*MEARTH/(DRESSINGPROPS[DRESSINGKOIS[k]['kic']]['Mstar']*MSUN))
    ALLPERIODS = np.array(ALLPERIODS)
    ALLRPS = np.array(ALLRPS)
    ALLKOIS = np.array(ALLKOIS)
    ALLMASSRATIOS = np.array(ALLMASSRATIOS)

    if USE_Q1Q8:
        MAXP = 90
    elif USE_Q1Q12:
        MAXP = 90
    else:
        MAXP = 50


    w = np.where(ALLPERIODS < MAXP)
    ALLPERIODS = ALLPERIODS[w]
    ALLRPS = ALLRPS[w]
    ALLKOIS = ALLKOIS[w]

except:
    if USE_Q1Q8:
        MAXP = 90
    elif USE_Q1Q12:
        MAXP = 90
    else:
        MAXP = 50

    print 'warning: ALLPERIODS, ALLRPS, etc. not setup.'
        

MAXP = 150

#DRESSINGKOIS: P,Teff,Rstar,arstar,b,kic,rp,Ep_rp,En_rp,rprstar,t0,F,Ep_F,En_F

STAROBSPROPS = np.recfromtxt('dressingkic_starobsprops.txt',names=True)
NOBSERVED = (STAROBSPROPS.Tobs > 0).sum()
#NOBSERVED = len(DRESSINGPROPS.keys())

data = pt.parsetxt('batalhaSNRs.csv',delimiter=',',comments=True)
BATALHASNRS = {}
for i in data.keys():
    m = re.search('K(\d\d\d\d\d\.\d\d)',data[i]['kepoi_name'])
    BATALHASNRS['KOI%.2f' % float(m.group(1))] = data[i]['koi_model_snr']

def write_etas(fname='etadiscs.txt',etafn=kfpp.SNRramp,simple=False):
    fout = open(fname,'w')
    fout.write('koi ptrans etadisc\n')
    for k in ALLKOIS:
        fout.write('%s %.3f %.3f\n' % (k,koi_transprob(k),koi_etadisc(k,etafn=etafn,simple=simple)))
    fout.close()

try:
    ETAS = pt.parsetxt('etadiscs.txt')
    if set(ETAS.keys()) != set(ALLKOIS):
        raise IOError
except IOError:
    pass
    #write_etas()

def koi_provenance(koi):
    if USE_Q1Q8:
        return KOIS_Q1Q8[koi]['provenance']
    elif USE_Q1Q12:
        return KOIS_Q1Q12[koi]['provenance']

def etafn_q1q8(ramp=(6,10)):
    return 

def etafn_q1q12(ramp=(6,10)):
    return

def koi_mp(koi,comp='earth'):
    return MofR(koi_rp(koi),comp=comp)

def koi_mstar(koi,err=False):
    try:
        if koi in BADKOIS:
            raise KeyError
        M = DRESSINGKOIS[koi]['Mstar']
        dM = (DRESSINGPROPS[DRESSINGKOIS[koi]['kic']]['En_Mstar'] + \
              DRESSINGPROPS[DRESSINGKOIS[koi]['kic']]['En_Mstar'])/2
    except KeyError:
        M = MUIRHEADKOIS[k]['M']
        dM = MUIRHEADKOIS[k]['e_M']
    if err:
        return M,dM
    else:
        return M

def koi_rp(koi,err=False):
    if USE_Q1Q8:
        rp = KOIS_Q1Q8[koi]['rp']
        drp = (KOIS_Q1Q8[koi]['En_rp'] + KOIS_Q1Q8[koi]['Ep_rp'])/2
    elif USE_Q1Q12:
        rp = KOIS_Q1Q12[koi]['rp']
        drp = (KOIS_Q1Q12[koi]['En_rp'] + KOIS_Q1Q12[koi]['Ep_rp'])/2
    else:
        try:
            if koi in BADKOIS:
                raise KeyError
            rp = DRESSINGKOIS[koi]['rp']
            drp = (DRESSINGKOIS[koi]['En_rp'] + DRESSINGKOIS[koi]['Ep_rp'])/2
        except KeyError:
            rp = MUIRHEADKOIS[k]['Rp']
            drp = MUIRHEADKOIS[k]['e_Rp']
    if err:
        return rp,drp
    else:
        return rp

def isoa(mp,ms,sigma0=26,gamma=-1,a0=5):
    return (((mp*MEARTH)**(2./3)*(3*ms*MSUN)**(1./3)/(16*np.pi*sigma0))**(1./(2+gamma)))*((a0*AU)**(gamma))/AU

def koi_isoa(koi,sigma0=26,gamma=-1,a0=5,err=False,errval=0.2,comp='earth'):
    """
    """
    mp = MofR(koi_rp(koi),comp=comp)
    ms = koi_mstar(koi)
    a = isoa(mp,ms,sigma0,gamma,a0)
    if err:
        return a,errval*a
    else:
        return a

def koi_massratio(koi,err=False,unc=0.2,comp='earth'):
    try:
        if koi in BADKOIS:
            raise KeyError
        massrat = MofR(DRESSINGKOIS[koi]['rp'],comp)*MEARTH/(DRESSINGPROPS[DRESSINGKOIS[koi]['kic']]['Mstar']*MSUN)
    except KeyError:
        massrat = MofR(MUIRHEADKOIS[k]['Rp'],comp)*MEARTH/(MUIRHEADKOIS[k]['M']*MSUN)
    if err:
        return massrat,massrat*unc
    else:
        return massrat

def semimajor(P,mstar=1):
    """P in days, mstar in Solar masses, returns a in AU
    """
    return ((P*DAY/2/np.pi)**2*G*mstar*MSUN)**(1./3)/AU

def setup_q1q8(infile='kois.csv',nophil=NOPHIL,outfile='kois_q1q8.pkl'):
    setup_kois(infile,nophil,outfile)

def setup_q1q12(infile='kois_q1q12.csv',nophil=NOPHIL,outfile='kois_q1q12.pkl'):
    setup_kois(infile,nophil,outfile)

def setup_kois(infile='kois.csv',nophil=NOPHIL,outfile='kois_q1q8.pkl'):
    data = np.recfromcsv(infile,names=True)
    KOIs = {}
    for i,kic in enumerate(data.kepid):
        if kic in DRESSINGPROPS:
            koi = data.kepoi_name[i]
            m = re.search('K(\d\d\d\d\d\.\d\d)',koi)
            if m:
                koi = 'KOI%.2f' % float(m.group(1))
            if data.koi_disposition[i] != 'FALSE POSITIVE' and data.koi_pdisposition[i] != 'FALSE POSITIVE':
                d = {}
                d['P'] = data.koi_period[i]
                d['b'] = data.koi_impact[i]
                try:
                    if nophil:
                        raise KeyError
                    if MUIRHEADKOIS[koi]['Teff'] > 3800 and koi not in SWIFTPROPS:
                        raise KeyError
                    if USESWIFT:
                        try:
                            d['Rstar'] = SWIFTPROPS[koi]['radius']
                            d['Teff'] = SWIFTPROPS[koi]['Teff']
                            a = semimajor(d['P'],SWIFTPROPS[koi]['mass'])*AU
                            d['arstar'] = a/(d['Rstar']*RSUN)
                            d['rp'] = SWIFTPROPS[koi]['Rp']
                            d['En_rp'] = SWIFTPROPS[koi]['e_Rp']
                            d['Ep_rp'] = SWIFTPROPS[koi]['e_Rp']
                            d['rprstar'] = SWIFTPROPS[koi]['RpRs']
                            d['star_provenance'] = 'swift'
                            d['fit_provenance'] = 'swift'
                            d['b'] = SWIFTPROPS[koi]['impact']
                        except:
                            print 'problem with Swift params for %s?  defaulting to muirhead params' % koi

                            d['Rstar'] = MUIRHEADKOIS[koi]['R']
                            d['Teff'] = MUIRHEADKOIS[koi]['Teff']
                            d['arstar'] = MUIRHEADKOIS[koi]['arstar']
                            d['rp'] = MUIRHEADKOIS[koi]['Rp']
                            d['En_rp'] = MUIRHEADKOIS[koi]['e_Rp']
                            d['Ep_rp'] = MUIRHEADKOIS[koi]['e_Rp']
                            d['rprstar'] = d['rp']*REARTH/(d['Rstar']*RSUN)
                            d['star_provenance'] = 'muirhead'
                            d['fit_provenance'] = 'kepler'
                            if np.isnan(d['rp']):
                                raise KeyError
                    else:
                        if MUIRHEADKOIS[koi]['Teff'] > 3800:
                            raise KeyError
                        d['Rstar'] = MUIRHEADKOIS[koi]['R']
                        d['Teff'] = MUIRHEADKOIS[koi]['Teff']
                        d['arstar'] = MUIRHEADKOIS[koi]['arstar']
                        d['rp'] = MUIRHEADKOIS[koi]['Rp']
                        d['En_rp'] = MUIRHEADKOIS[koi]['e_Rp']
                        d['Ep_rp'] = MUIRHEADKOIS[koi]['e_Rp']
                        d['rprstar'] = d['rp']*REARTH/(d['Rstar']*RSUN)
                        d['star_provenance'] = 'muirhead'
                        d['fit_provenance'] = 'kepler'
                        if np.isnan(d['rp']):
                            raise KeyError
                except KeyError:
                    #figure out where transit fits are coming from here....
                    print 'using Dressing props for %s' % koi
                    d['Rstar'] = DRESSINGPROPS[kic]['Rstar']
                    d['Teff'] = DRESSINGPROPS[kic]['Teff']
                    d['star_provenance'] = 'dressing'
                    try:
                        d['arstar'] = DRESSINGKOIS[koi]['arstar']
                        d['rp'] = DRESSINGKOIS[koi]['rp']
                        d['rprstar'] = DRESSINGKOIS[koi]['rprstar']
                        d['En_rp'] = DRESSINGKOIS[koi]['En_rp']
                        d['Ep_rp'] = DRESSINGKOIS[koi]['Ep_rp']
                        d['fit_provenance'] = 'dressing'
                    except KeyError:
                        a = semimajor(d['P'],DRESSINGPROPS[kic]['Mstar'])*AU
                        d['arstar'] = a/(d['Rstar']*RSUN)
                        d['rp'] = data.koi_ror[i]*(d['Rstar']*RSUN)/REARTH
                        d['rprstar'] = data.koi_ror[i]
                        d['En_rp'] = DRESSINGPROPS[kic]['En_Rstar']/d['Rstar'] * d['rp']
                        d['Ep_rp'] = DRESSINGPROPS[kic]['Ep_Rstar']/d['Rstar'] * d['rp']
                        d['fit_provenance'] = 'kepler'

                if np.isnan(d['rp']):
                    print koi,d
                d['SNR'] = data.koi_model_snr[i]
                d['kic'] = kic
                KOIs[koi] = d
    f = open(outfile,'wb')
    pickle.dump(KOIs,f)
    f.close()

try:
    f = open('kois_q1q8.pkl','rb')
    d = pickle.load(f)
    KOIS_Q1Q8 = ku.make_KOIdict(d)
except:
    setup_q1q8()
    f = open('kois_q1q8.pkl','rb')
    d = pickle.load(f)
    KOIS_Q1Q8 = ku.make_KOIdict(d)
try:
    f = open('kois_q1q12.pkl','rb')
    d = pickle.load(f)
    KOIS_Q1Q12 = ku.make_KOIdict(d)
except:
    setup_q1q12()
    f = open('kois_q1q12.pkl','rb')
    d = pickle.load(f)
    KOIS_Q1Q12 = ku.make_KOIdict(d)


#for k in NEWRADII.keys():
#    KOIS_Q1Q12[k]['rp'] = NEWRADII[k]


def plot_snrcompare(fig=None,color='k',maxsnr=50,**kwargs):
    plu.setfig(fig)
    for k in ALLKOIS:
        plt.plot(BATALHASNRS[k],SNRprops(k)['SNR'],'+',color=color)
    plt.xlim((0,maxsnr))
    plt.ylim((0,maxsnr))
    x = np.arange(50)
    plt.plot(x,x,'k:')
    plt.xlabel('SNR [Batalha+ (2012)]')
    plt.ylabel('SNR [this work]')

def eta_snrthresh(snr,thresh=7.1):
    snr = np.atleast_1d(snr)
    effs = snr*0
    effs[np.where(snr >= thresh)] = 1.
    if np.size(effs)==1:
        return effs[0]
    else:
        return effs

def write_rkdes(ramp=SNRRAMP,folder=None,pfixedfolder=None):
    rampfn = kfpp.SNRrampfn(*ramp)
    if folder is None:
        if USE_Q1Q8:
            folder = 'koi_snrfns_q1q8'
        elif USE_Q1Q12:
            folder = 'koi_snrfns_q1q12'
        else:
            folder = 'koi_snrfns_batalha'
    if pfixedfolder is None:
        if USE_Q1Q8:
            pfixedfolder = 'koi_snrfns_q1q8_pfixed'
        elif USE_Q1Q12:
            pfixedfolder = 'koi_snrfns_q1q12_pfixed'
        else:
            pfixedfolder = 'koi_snrfns_batalha_pfixed'

    rkde = KOI_radiusKDE(etafn=rampfn,folder=folder,recalc=True)
    rkde.write_table('rkde.txt')

    rkdesimple = KOI_radiusKDE(etafn=rampfn,simple=True,folder=pfixedfolder,recalc=True)
    rkdesimple.write_table('rkdesimple.txt')

    rkdesimplethresh = KOI_radiusKDE(simple=True,etafn=eta_snrthresh,recalc=True,
                                         folder=pfixedfolder)
    rkdesimplethresh.write_table('rkdesimplethresh.txt')
    

def simple_Nstar(koi,etafn=None):
    if etafn is None:
        etafn = eta_snrthresh
    d = STAROBSPROPS
    props = SNRprops(koi)
    newsnrs = props['SNR']*\
        (d.Rstar/props['Rstar'])**(-2)*\
        (d.CDPP/props['CDPP'])**(-1)*\
        (d.Tobs/props['Tobs'])**(1./2)
    #return (newsnrs > thresh).sum()
    return (etafn(newsnrs[np.where(~np.isnan(newsnrs))])).sum()

def koi_transprob(koi):
    try:
        if USE_Q1Q8:
            d = KOIS_Q1Q8[koi]
        elif USE_Q1Q12:
            d = KOIS_Q1Q12[koi]
        else:
            if koi in BADKOIS:
                raise KeyError
            d = DRESSINGKOIS[koi]
    except KeyError:
        if USE_Q1Q8:
            print 'should not happen...muirhead props for %s' % koi
        elif USE_Q1Q12:
            print 'should not happen...muirhead props for %s' % koi
        d = MUIRHEADKOIS[koi]
    return (1./d['arstar'])

def koi_etadisc(koi,etafn=kfpp.SNRramp,folder='koi_snrfns_q1q6',simple=False,maxq=MAXQ,recalc=True):
    if not recalc:
        return ETAS[koi]['etadisc']
    if simple:
        props = SNRprops(koi,maxq=maxq)
        eta = float(simple_Nstar(koi,etafn=etafn))/NOBSERVED
        #if etafn is not None:
        #    eta *= etafn(props['SNR'])
        return eta
    else:
        snrfn = koi_SNRfn(koi,folder=folder)
        return snrfn.integrate_efficiency(etafn=etafn)

def koi_weight(koi,etafn=None,verbose=False,folder='koi_snrfns_q1q6'):
    transprob = koi_transprob(koi)
    etadisc = koi_etadisc(koi,etafn,folder=folder)
    if verbose:
        print 'transit probability: %.3f, eta_disc = %.2f' % (transprob,etadisc)
    return 1/(transprob*etadisc)

def SNRprops(koi,maxq=MAXQ,batalha=True):
    koi = ku.koiname(koi)
    try:
        if USE_Q1Q8:
            d = KOIS_Q1Q8[koi]
        elif USE_Q1Q12:
            d = KOIS_Q1Q12[koi]            
        else:
            d = DRESSINGKOIS[koi]
        kic = d['kic']
        Rp = d['rp']
        period = d['P']
        Rstar = d['Rstar']
        duration = transit_T14(koi)
    except KeyError:
        raise
        if USE_Q1Q8:
            print 'this should not happen? (SNRprops going to MUIRHEAD for %s)' % koi
        elif USE_Q1Q12:
            print 'this should not happen? (SNRprops going to MUIRHEAD for %s)' % koi
        d = MUIRHEADKOIS[koi]
        kic = d['kic']
        Rp = d['Rp']
        period = d['period']
        Rstar = d['R']
        duration = d['duration']
    if USE_Q1Q8:
        SNR = d['SNR']
    elif USE_Q1Q12:
        SNR = d['SNR']
    elif batalha:
        SNR = BATALHASNRS[koi]
    else:
        SNR = kfpp.totalSNR(koi,maxq=maxq)
    Tobs = kfpp.days_observed(kic,maxq=maxq)
    CDPP = kfpp.median_CDPP(kic,maxq=maxq)
    return dict(SNR=SNR,Tobs=Tobs,CDPP=CDPP,Rstar=Rstar,Rp=Rp,
                duration=duration,period=period)

def generic_SNRfn(SNR=20.,P=10.,Rp=1.,Rs=0.5,duration=2.,CDPP=200,Tobs=500,
                  kdewidth=0.15,N=1e4,maxq=MAXQ,SNRmax=50,kois=None,remake=False):
    """Do not change these defaults or the fn will be wrong...[should connect these to the saved file...]
    """
    kde = logper_kde(kdewidth,kois=kois)
    props = dict(SNR=SNR,period=P,Rp=Rp,Rstar=Rs,duration=duration,CDPP=CDPP,Tobs=Tobs)
    snrs,vals = np.loadtxt('generic_snrfn.txt',unpack=True)
    fn = interpolate(snrs,vals,s=0)
    if remake:
        return occ.make_SNRfn(props,kde,SNRmax=SNRmax,N=N)
    else:
        return occ.SNRfn(fn,props,SNRmax,DRESSINGPROPS.keys(),kde,maxq)

def transit_T14(koi):
    koi = ku.koiname(koi)
    if koi in BADKOIS or koi not in DRESSINGKOIS or koi not in KOIS_Q1Q8 or koi not in KOIS_Q1Q12:
        return kfpp.KOIDATA[koi]['koi_duration']

    if koi in SWIFTPROPS:
        if USE_Q1Q12:
            d = KOIS_Q1Q12[koi]
        elif USE_Q1Q8:
            d = KOIS_Q1Q8[koi]
    else:
        d = DRESSINGKOIS[koi]

    k = d['rprstar']
    inc = np.pi/2 - d['b']/d['arstar']
    #print d['P']*DAY/np.pi/3600
    #print np.arcsin(1./d['arstar'] * np.sqrt((1+k)**2 - d['b']**2)/np.sin(inc))
    return d['P']*DAY/np.pi * np.arcsin(1./d['arstar'] * np.sqrt((1+k)**2 - d['b']**2)/np.sin(inc))/3600

def makeall_snrfns(kois=None,folder='koi_snrfns_q1q12',overwrite=False,**kwargs):
    i=1
    if kois is None:
        kois = ALLKOIS
    for k in kois:
        print '%i of %i:' % (i,len(kois))
        try:
            filename = '%s/%s.txt' % (folder,k)
            if os.path.exists(filename) and not overwrite:
                continue
            snrfn = koi_SNRfn(k,remake=True,**kwargs)
            snrfn.save_fn('%s/%s.txt' % (folder,k))
        except:
            raise
            print 'skipped %s' % k
        i += 1

def koi_SNRfn(koi,snrfn=None,folder='koi_snrfns_q1q12',maxq=MAXQ,SNRmax=50,
              remake=False,kdewidth=0.15,kois=None,N=1e4,Pfixed=False,
              simple=False,pdist=None):
    koi = ku.koiname(koi)
    props = SNRprops(koi,maxq)
    if remake:
        if pdist is None:
            pdist = logper_kde(kdewidth,kois=kois)
        return occ.make_SNRfn(props,pdist,SNRmax=SNRmax,N=N,Pfixed=Pfixed,simple=simple)
    else:
        if snrfn is None:
            if pdist is None:
                kde = logper_kde(kdewidth,kois=kois)
            else:
                kde = pdist
            return occ.SNRfn('%s/%s.txt' % (folder,koi),props,SNRmax,
                             DRESSINGPROPS.keys(),kde,maxq)
        else:
            return occ.modify_SNRfn(snrfn,props,SNRmax)
             
def logper_kde(width=0.15,kois=None,rbin=(0,np.inf),pmax=MAXP,raw=False,**kwargs):
    if kois is None:
        kois = ALLKOIS

    logpers = []
    transprobs = []
    for k in kois:
        try:
            if USE_Q1Q8:
                d = KOIS_Q1Q8[k]
            elif USE_Q1Q12:
                d = KOIS_Q1Q12[k]
            else:
                if k in BADKOIS:
                    raise KeyError
                d = DRESSINGKOIS[k]
            if d['rp'] < rbin[0] or d['rp'] > rbin[1]:
                continue
            if d['P'] < pmax:
                logpers.append(np.log10(d['P']))
                transprobs.append(1./d['arstar'])
        except KeyError:
            if USE_Q1Q8:
                print 'This should not happen.  going to muirhead params for %s.' % k
            elif USE_Q1Q12:
                print 'This should not happen.  going to muirhead params for %s.' % k
            d = MUIRHEADKOIS[k]
            if d['Rp'] < rbin[0] or d['Rp'] > rbin[1]:
                continue
            logpers.append(np.log10(d['period']))
            transprobs.append(1./d['arstar'])

    logpers = np.array(logpers)
    transprobs = np.array(transprobs)

    if raw:
        return ModifiedKDE(logpers,weights=None,widths=width,norm=NOBSERVED,minval=np.log10(0.5),maxval=np.log10(pmax),**kwargs)
    else:
        return ModifiedKDE(logpers,weights=1./transprobs,widths=width,norm=NOBSERVED,minval=np.log10(0.5),maxval=np.log10(pmax),**kwargs)


class ModifiedKDE(object):
    def __init__(self,data,weights=None,widths=None,norm=None,minval=None,maxval=None,normed=True):
        """A modified 1d kernel density estimator, allowing for weights for each data point

        if used in the context of a survey, norm should be the number of total stars surveyed
        """
        self.data = data
        self.N = len(data)
        if norm is None:
            norm = 1
        self.norm = norm
        self.normed = normed
        
        if weights is None:
            weights = np.ones(data.shape)
        self.weights = weights

        if widths is None:
            widths = self.N**(-1./5)*np.ones(data.shape)
        elif type(widths) in [type(1),type(0.1)]:
            widths = widths*np.ones(data.shape)
        self.widths = widths

        if minval is None:
            minval = data.min() - 3*widths[np.argmin(data)]
        if maxval is None:
            maxval = data.max() + 3*widths[np.argmax(data)]
        self.minval = minval
        self.maxval = maxval
        self._setfns()

    def adjust_width(self,factor):
        self.set_width(self,widths*factor)

    def set_width(self,widths):
        if type(widths) in [type(1),type(.1)]:
            self.widths = np.ones(self.data.shape)*widths
        else:
            self.widths = widths
        self._setfns()

    def _setfns(self):
        vals = np.linspace(self.minval,self.maxval,1000)
        tot = self.evaluate(vals)
        #print vals,tot
        #print 'setting functions...'
        if self.normed:
            self.norm *= trapz(tot,vals) #might not be right
        self.pdf = interpolate(vals,tot,s=0)
        self.cdf = interpolate(vals,tot.cumsum()/tot.cumsum().max(),s=0)

    def evaluate(self,x):
        x = np.atleast_1d(x)
        tot = x*0
        for d,sig,w in zip(self.data,self.widths,self.weights):
            tot += 1./np.sqrt(2*np.pi*sig**2)*np.exp(-(x-d)**2/(2*sig**2))*w
        tot /= self.norm
        if np.size(tot)==1:
            return tot[0]
        else:
            return tot

    def bootstrap_bias(self,N=1000,npts=500):
        xs = np.linspace(self.minval,self.maxval,npts)
        norm = quad(self,self.minval,self.maxval)
        tots = np.zeros(npts)
        for i in np.arange(N):
            pass
            

    def bootstrap(self,N=1000,npts=500,use_pbar=True,return_vals=False):
        xs = np.linspace(self.minval,self.maxval,npts)
        tots = np.zeros((N,npts))
        if use_pbar:
            widgets = ['calculating bootstrap variance: ',Percentage(),' ',
                       Bar(marker=RotatingMarker()),' ',ETA()]
            pbar = ProgressBar(widgets=widgets,maxval=N)
            pbar.start()
        for i in np.arange(N):
            inds = rand.randint(self.N,size=self.N)
            new = ModifiedKDE(self.data[inds],self.weights[inds],
                              self.widths[inds],norm=self.norm,
                              minval=self.minval,maxval=self.maxval,normed=self.normed)
            tots[i,:] = new(xs)
            if use_pbar and i % 10==0:
                pbar.update(i)
        if use_pbar:
            pbar.finish()
        sorted = np.sort(tots,axis=0)
        pvals = sorted[-N*16/100,:] #84th pctile pts
        mvals = sorted[N*16/100,:] #16th pctile pts
        self.uncfn_p = interpolate(xs,pvals,s=0)
        self.uncfn_m = interpolate(xs,mvals,s=0)
        if return_vals:
            return xs,tots

    def __call__(self,x):
        x = np.atleast_1d(x)
        vals = self.pdf(x)
        vals = np.atleast_1d(vals)
        vals[np.where((x < self.minval) | (x > self.maxval))] = 0
        if np.size(vals)==1:
            return vals[0]
        else:
            return vals

    def plot(self,fig=None,log=False,scale=1.,label=None,uncs=False,lines=True,xtickfmt=None,
             unc_color='k',unc_alpha=0.1,minval=None,maxval=None,**kwargs):
        plu.setfig(fig)
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        xvals = np.linspace(minval,maxval,1000)
        if log:
            plt.semilogx(10**xvals,scale*self(xvals),label=label,**kwargs)
            if lines:
                for d,w in zip(self.data,self.weights):
                    plt.axvline(10**d,color='r',lw=1,ymax=max(0.01,w*0.0015))
            if uncs:
                if self.uncfn_p is None:
                #if not hasattr(self,'uncfn_p'):
                    self.bootstrap()
                hi = scale*self.uncfn_p(xvals)
                lo = scale*self.uncfn_m(xvals)
                plt.fill_between(10**xvals,hi,lo,color=unc_color,alpha=unc_alpha)
                    
            plt.xlim((10**minval,10**maxval))
            ax = plt.gca()
            if xtickfmt is not None:
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(xtickfmt))
        else:
            plt.plot(xvals,scale*self(xvals),label=label,**kwargs)
            if lines:
                for r,w in zip(self.data,self.weights):
                    plt.axvline(r,color='r',lw=1,ymax=max(0.01,w*0.0015))
            if uncs:
                if self.uncfn_p is None:
                #if not hasattr(self,'uncfn_p'):
                    self.bootstrap()
                hi = scale*self.uncfn_p(xvals)
                lo = scale*self.uncfn_m(xvals)
                plt.fill_between(xvals,hi,lo,color=unc_color,alpha=0.2)
                
            plt.xlim((minval,maxval))
            
        plt.ylabel('Probability Density')
        plt.yticks([])

    def resample(self,N):
        u = rand.random(size=N)
        vals = np.linspace(self.minval,self.maxval,1e4)
        ys = self.cdf(vals)
        inds = np.digitize(u,ys)
        return vals[inds]
        pass
        

class LogperKDE(ModifiedKDE):
    def __init__(self,kois=None,width=None):
        pass

    def plot(self,fig=None,**kwargs):
        ModifiedKDE.plot(self,fig,**kwargs)
        plt.xlabel('log $P$')
        plt.ylabel('Probability Density')
        plt.yticks([])


class KOI_massratKDE_old(ModifiedKDE):
    def __init__(self,tablefile=None,kois=None,folder='koi_snrfns_batalha',etafn=kfpp.SNRramp,
                 simple=False,maxval=None,minval=None,width=0.05,log=True,comp='earth'):
        self.log = log
        if tablefile is not None:
            kois = np.loadtxt(tablefile,usecols=(0,),dtype='str')
            massrats,dmassrats,transprobs,etadiscs = np.loadtxt(tablefile,usecols=(1,2,3,4),unpack=True)
        else:
            if kois is None:
                kois = ALLKOIS

            massrats = []
            dmassrats = []
            transprobs = []
            etadiscs = []
            for k in kois:
                eta = koi_etadisc(k,etafn,folder=folder,simple=simple)
                trprob = koi_transprob(k)
                if np.isnan(eta):
                    print 'eta_disc is nan for %s (skipping)' % k
                    continue
                massrat,dmassrat = koi_massratio(k,error=True)
                if log:
                    massrat = np.log10(massrat)
                    if width is not None:
                        dmassrat = width*massrat
                    else:
                        dmassrat = np.log10(dmassrat)
                massrats.append(massrat)
                dmassrats.append(dmassrat)
                transprobs.append(trprob)
                etadiscs.append(eta)
        self.kois = kois
        self.transprobs = np.array(transprobs)
        self.etadiscs = np.array(etadiscs)

        ModifiedKDE.__init__(self,np.array(massrats),weights=(1./(self.transprobs*self.etadiscs)),
                             widths=np.array(dmassrats),maxval=maxval,minval=minval,
                             norm=NOBSERVED,normed=False)

    def plot(self,**kwargs):
        ModifiedKDE.plot(self,log=self.log,**kwargs)
        plt.xlabel('Mass Ratio [$M_p/M_\star$]')

    def write_table(self,filename='mkde.txt'):
        fout = open(filename,'w')
        fout.write('#KOI massrat dmassrat pr_trans eta_disc weight\n')
        for k,m,dm,ptr,eta in zip(self.kois,self.data,self.widths,self.transprobs,self.etadiscs):
            fout.write('%-12s %.2f %.2f %.3f %.3f %.2f\n' % (k,m,dm,ptr,eta,1./(ptr*eta)))
        fout.close()

class KOIKDE(ModifiedKDE):
    def __init__(self,propfn,tablefile=None,kois=None,folder='koi_snrfns_batalha',etafn=kfpp.SNRramp,
                 simple=False,maxval=None,width=None,fracwidth=None,log=False,recalc=True,widthfactor=1.):
        if tablefile is not None:
            kois = np.loadtxt(tablefile,usecols=(0,),dtype='str')
            data,ddata,transprobs,etadiscs = np.loadtxt(tablefile,usecols=(1,2,3,4),unpack=True)
        else:
            if kois is None:
                kois = ALLKOIS

            data = []
            ddata = []
            transprobs = []
            etadiscs = []
            for k in kois:
                eta = koi_etadisc(k,etafn,folder=folder,simple=simple,recalc=recalc)
                trprob = koi_transprob(k)
                if np.isnan(eta):
                    print 'eta_disc is nan for %s (skipping)' % k
                    continue
                if width is None:
                    d,dd = propfn(k,err=True)
                else:
                    d = propfn(k,err=False)
                    dd = width
                if log:
                    d = np.log10(d)
                    #dd = np.log10(dd)
                #print d,dd,k
                data.append(d)
                ddata.append(dd)
                transprobs.append(trprob)
                etadiscs.append(eta)
        self.kois = kois
        self.transprobs = np.array(transprobs)
        self.etadiscs = np.array(etadiscs)
        self.log = log
        self.widthfactor = widthfactor

        ModifiedKDE.__init__(self,np.array(data),weights=(1./(self.transprobs*self.etadiscs)),
                             widths=np.array(ddata)*widthfactor,maxval=maxval,
                             norm=NOBSERVED,normed=False)

    def write_table(self,filename='kde.txt'):
        fout = open(filename,'w')
        fout.write('#KOI data unc pr_trans eta_disc weight\n')
        for k,m,dm,ptr,eta in zip(self.kois,self.data,self.widths,self.transprobs,self.etadiscs):
            fout.write('%-12s %.2f %.2f %.3f %.2f %.1f\n' % (k,m,dm,ptr,eta,1./(ptr*eta)))
        fout.close()

    def write_tex(self,filename='kde.tex',provenance=True):
        fout = open(filename,'w')
        for k,x,dx,ptr,eta in zip(self.kois,self.data,self.widths,self.transprobs,self.etadiscs):
            name = k
            if provenance:
                if USE_Q1Q8:
                    if KOIS_Q1Q8[k]['provenance'] == 'muirhead':
                        mark = 'a'
                    elif KOIS_Q1Q8[k]['provenance'] == 'dressing':
                        mark = 'b'
                elif USE_Q1Q12:
                    if KOIS_Q1Q12[k]['provenance'] == 'muirhead':
                        mark = 'a'
                    elif KOIS_Q1Q12[k]['provenance'] == 'dressing':
                        mark = 'b'
                else:
                    raise ValueError('no provenance for %s' % k)
                line = '%s\\tablenotemark{%s} & %.2f & %.2f & %.3f & %.2f & %.1f\\\\\n' % (name,mark,x,dx,ptr,eta,1./(ptr*eta))
            else:
                line = '%-12s & %.2f & %.2f & %.3f & %.2f & %.1f\\\\\n' % (name,x,dx,ptr,eta,1./(ptr*eta))
            fout.write(line)

        fout.close()

class KOI_massratKDE(KOIKDE):
    def __init__(self,tablefile=None,comp='earth',rmax=2.5,kois=ALLKOIS,**kwargs):
        #fix this with a "fnargs" keyword...
        def fn(r,comp=comp,**kwargs):
            return koi_massratio(r,comp=comp,**kwargs)
        newkois = []
        for k in kois:
            if koi_rp(k) <= rmax:
                newkois.append(k)
        KOIKDE.__init__(self,fn,log=True,kois=newkois,**kwargs)

    def plot(self,**kwargs):
        ModifiedKDE.plot(self,log=self.log,**kwargs)
        plt.xlabel('Mass Ratio [$M_p/M_\star$]')


class KOI_radiusKDE(KOIKDE):
    def __init__(self,*args,**kwargs):
        KOIKDE.__init__(self,koi_rp,*args,**kwargs)

    def plot(self,fig=None,rmax=4,lines=True,hist=False,histymax=None,histlabel=True,
             histcolor='b',return_histvals=False,rmin=0.25,histbins=None,
             label=None,host=None,uncs=False,**kwargs):
        plu.setfig(fig)
        if host is None and fig != 0 :
            host = host_subplot(111)#, axes_class=AA.Axes)
        
        ModifiedKDE.plot(self,fig=0,color='k',label=label,uncs=uncs,**kwargs)
        plt.ylabel('Planet Radius Distribution Function $\phi^{%i}_r$' % MAXP)
        plt.xlabel('Planet Radius [$R_e$]')
        plt.xlim((rmin,rmax))
        if lines:
            for r,w in zip(self.data,self.weights):
                plt.axvline(r,color='r',lw=1,ymax=max(0.01,w*0.0015))
        if hist:
            if histbins is None:
                histbins = np.logspace(np.log10(0.5),np.log10(4),7)
            rh = host.twinx()
            rh.axis['right'].toggle(all=True)
            #rh.yaxis.set_label_coords(1.08,0.25)
            #rh.set_ylabel('Occurrence Rate')
            vals = self.plot_hist(fig=0,ax=rh,color=histcolor,bins=histbins,**kwargs)
            if histymax is None:
                histymax = 2*max(vals)
            rh.set_ylim(ymax=histymax)
            rh.yaxis.label.set_color(histcolor)
            rh.set_yticks(np.arange(0.1,max(vals)+0.1,0.1))
            #rh.set_yticks([0.1,0.2,0.3,0.4,0.5])
            rh.tick_params(axis='y', colors=histcolor)
            rh.spines['right'].set_color(histcolor)
            #plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6])
            if histlabel:
                plt.annotate('Avg. # of planets per star, $P$ < %i days' % MAXP,
                             xy=(1.07,0.80),xycoords='axes fraction',
                             rotation=90,fontsize=14,color='b')
            if return_histvals:
                return rh,vals
            
    def plot_hist(self,fig=None,bins=np.logspace(np.log10(0.5),np.log10(4),7),color='k',ls='-',
                  ax=None,**kwargs):
        plu.setfig(fig)
        if ax is None:
            ax = plt.gca()
        inds = np.digitize(self.data,bins)
        vals = []
        for i in np.arange(len(bins)-1)+1:
            w = np.where(inds==i)
            val = self.weights[w].sum()/self.norm
            dbin = bins[i]-bins[i-1]
            xs = np.array([bins[i-1],bins[i]])
            ys = np.array([val,val])
            ax.plot(xs,ys,color=color,ls=ls,**kwargs)
            vals.append(val)
        return vals

    def write_table(self,filename='radfn_table.txt'):
        fout = open(filename,'w')
        fout.write('#KOI Rp dRp pr_trans eta_disc weight\n')
        for k,r,dr,ptr,eta in zip(self.kois,self.data,self.widths,self.transprobs,self.etadiscs):
            fout.write('%-12s %.2f %.2f %.3f %.2f %.1f\n' % (k,r,dr,ptr,eta,1./(ptr*eta)))
        fout.close()


class KOI_radiusKDE_old(ModifiedKDE):
    def __init__(self,tablefile=None,kois=None,folder='koi_snrfns_batalha',etafn=kfpp.SNRramp,
                 simple=False,maxval=None,width=None):
        if tablefile is not None:
            kois = np.loadtxt(tablefile,usecols=(0,),dtype='str')
            rps,drps,transprobs,etadiscs = np.loadtxt(tablefile,usecols=(1,2,3,4),unpack=True)
        else:
            if kois is None:
                kois = ALLKOIS

            rps = []
            drps = []
            transprobs = []
            etadiscs = []
            for k in kois:
                eta = koi_etadisc(k,etafn,folder=folder,simple=simple)
                trprob = koi_transprob(k)
                if np.isnan(eta):
                    print 'eta_disc is nan for %s (skipping)' % k
                    continue
                try:
                    if USE_Q1Q8:
                        rps.append(KOIS_Q1Q8[k]['rp'])
                        drps.append((KOIS_Q1Q8[k]['En_rp'] + KOIS_Q1Q8[k]['Ep_rp'])/2.)
                    elif USE_Q1Q12:
                        rps.append(KOIS_Q1Q12[k]['rp'])
                        drps.append((KOIS_Q1Q12[k]['En_rp'] + KOIS_Q1Q12[k]['Ep_rp'])/2.)
                    else:
                        rps.append(DRESSINGKOIS[k]['rp'])
                        drps.append((DRESSINGKOIS[k]['En_rp'] + DRESSINGKOIS[k]['Ep_rp'])/2.)
                except KeyError:
                    if USE_Q1Q8:
                        print 'this should not happen...muirhead props for %s' % k
                    elif USE_Q1Q12:
                        print 'this should not happen...muirhead props for %s' % k
                    rps.append(MUIRHEADKOIS[k]['Rp'])
                    drps.append(MUIRHEADKOIS[k]['e_Rp'])
                transprobs.append(trprob)
                etadiscs.append(eta)
        self.kois = kois
        self.transprobs = np.array(transprobs)
        self.etadiscs = np.array(etadiscs)

        ModifiedKDE.__init__(self,np.array(rps),weights=(1./(self.transprobs*self.etadiscs)),
                             widths=np.array(drps),maxval=maxval,
                             norm=NOBSERVED,normed=False)

    def plot(self,fig=None,rmax=4,lines=True,hist=False,histymax=None,histlabel=True,
             histcolor='b',return_histvals=False,rmin=0.25,
             label=None,host=None,uncs=False,**kwargs):
        plu.setfig(fig)
        if host is None and fig != 0 :
            host = host_subplot(111)#, axes_class=AA.Axes)
        
        ModifiedKDE.plot(self,fig=0,color='k',label=label,uncs=uncs,**kwargs)
        plt.ylabel('Planet Radius Distribution Function $\phi^{50}_r$')
        plt.xlabel('Planet Radius [$R_e$]')
        plt.xlim((rmin,rmax))
        if lines:
            for r,w in zip(self.data,self.weights):
                plt.axvline(r,color='r',lw=1,ymax=max(0.01,w*0.0015))
        if hist:
            rh = host.twinx()
            rh.axis['right'].toggle(all=True)
            #rh.yaxis.set_label_coords(1.08,0.25)
            #rh.set_ylabel('Occurrence Rate')
            vals = self.plot_hist(fig=0,ax=rh,color=histcolor,**kwargs)
            if histymax is None:
                histymax = 2*max(vals)
            rh.set_ylim(ymax=histymax)
            rh.yaxis.label.set_color(histcolor)
            rh.set_yticks([0.1,0.2,0.3,0.4,0.5])
            rh.tick_params(axis='y', colors=histcolor)
            rh.spines['right'].set_color(histcolor)
            #plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6])
            if histlabel:
                plt.annotate('Avg. # of planets per star, $P$ < %i days' % MAXP,
                             xy=(1.07,0.80),xycoords='axes fraction',
                             rotation=90,fontsize=14,color='b')
            if return_histvals:
                return rh,vals
            
    def plot_hist(self,fig=None,bins=np.logspace(np.log10(0.5),np.log10(4),7),color='k',ls='-',
                  ax=None,**kwargs):
        plu.setfig(fig)
        if ax is None:
            ax = plt.gca()
        inds = np.digitize(self.data,bins)
        vals = []
        for i in np.arange(len(bins)-1)+1:
            w = np.where(inds==i)
            val = self.weights[w].sum()/self.norm
            dbin = bins[i]-bins[i-1]
            xs = np.array([bins[i-1],bins[i]])
            ys = np.array([val,val])
            ax.plot(xs,ys,color=color,ls=ls,**kwargs)
            vals.append(val)
        return vals

    def write_table(self,filename='radfn_table.txt'):
        fout = open(filename,'w')
        fout.write('#KOI Rp dRp pr_trans eta_disc weight\n')
        for k,r,dr,ptr,eta in zip(self.kois,self.data,self.widths,self.transprobs,self.etadiscs):
            fout.write('%-12s %.2f %.2f %.3f %.2f %.1f\n' % (k,r,dr,ptr,eta,1./(ptr*eta)))
        fout.close()

def plot_perdist(fig=None):
    kde = logper_kde()
    kde.plot(fig=fig,log=True,lw=3)
    plt.xlabel('$P$ [days]')
    plt.xticks([1,3,10,30,60])
    plt.savefig('perdist.png')

def SNR_of_P(orig,Ps):
    return orig['SNR']*(Ps/orig['period'])**(-1./2)

def plot_koi_snrofp(rkde,rbin=(0,1.0),width=0.2,fig=None):
    """other example was 2238.01, 952.05
    """

    plu.setfig(fig)
    #plt.plot(10**logps,newsnrs,'k')

    #kde = logper_kde(rbin=rbin,width=width,raw=True,normed=True)
    
    w = np.where((rkde.survey.detections['Rp'] < rbin[1]) & 
                 (rkde.survey.detections['Rp'] > rbin[0]))
    logps = np.log10(rkde.survey.detections['P'][w])
    kde = occ.WeightedKDE(logps,weights=None,widths=width,normed=True)

    logp_grid = np.arange(np.log10(0.5),np.log10(MAXP+10),0.01)

    #kdenorm = kde(logp_grid).max() * 3
    kdenorm = 1

    plt.plot(10**logp_grid,kde(logp_grid)/kdenorm,'k')
    plt.xscale('log')
    plt.xlim((0.5,MAXP+10))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))
    plt.xticks([1,3,10,30,100])
    plt.yticks([])
    plt.ylabel('Probability Density')
    #plt.ylabel('Signal-to-Noise Ratio')
    plt.xlabel('Orbital Period [days]')
    #plt.ylim(ymax=1.3*kde(logps).max()/kde)

    #plt.annotate('$SNR=%.1f$\n$P = %.2f$ d\n$R_p = %.2f R_e$\n$R_s = %.2f R_\odot$' %
    #             (props['SNR'],props['period'],props['Rp'],props['Rstar']),
    #             xy=(props['period'],props['SNR']),xytext=(50,8),
    #             textcoords='offset points',arrowprops=dict(arrowstyle="->",lw=2),
    #             fontsize=14,bbox=dict(boxstyle='round',fc='w'))

    plt.annotate('Smallest planets\n($R_p < %.1f R_e$)' % (rbin[1]),
                 xy=(1,kde(0)/kdenorm),ha='center',
                 xytext=(0.25,0.15),textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->',lw=2),
                 fontsize=16)
    plt.annotate('All Cool KOIs',
                 xy=(0.57,0.3),xycoords='axes fraction',
                 bbox=dict(boxstyle='round',fc='w',color='gray'),
                 alpha=0.5,
                 fontsize=16)

    #kdeall = logper_kde(raw=True,normed=True,width=0.2)
    kdeall = occ.WeightedKDE(np.log10(rkde.survey.detections['P']),weights=None,widths=width,normed=True)

    #kdeallnorm = kdeall(logp_grid).max() * 2
    kdeallnorm = 1
    #plt.plot(10**logps,kdeall(logps),'k',lw=3,zorder=1)
    plt.fill_between(10**logp_grid,kdeall(logp_grid)/kdeallnorm,logp_grid*0,color='k',alpha=0.15)

    #plt.ylim(ymax=plt.axis()[3]*1.2)
    plt.ylim(ymax=plt.axis()[3]*1.5)

    plt.axvline(23,ls='--',color='r') # SNR = 7.1 for 990d observation, CDPP=344, b=0, Rp=1,Rs=0.5

    x,y = (23,0.8*plt.axis()[3])
    plt.annotate('SNR = 7.1 for $R_p = 1 R_\oplus$, $R_\star = 0.5 R_\odot$\n(median $T_{obs}$ and CDPP)',
                 xy=(x,y),xytext=(0.6,0.95),textcoords='axes fraction',
                 color='r',fontsize=14,va='top',ha='right',
                 arrowprops=dict(arrowstyle='->',lw=2,color='r'))

    plt.savefig('pincomplete.png')
    plt.savefig('pincomplete.pdf')
    
def plot_koi_snrofp_old(kois=['KOI2036.02'],rbin=(0,1.0),width=0.2,fig=None):
    """other example was 2238.01, 952.05
    """

    logps = np.arange(np.log10(0.5),np.log10(MAXP+10),0.01)

    plu.setfig(fig)
    #plt.plot(10**logps,newsnrs,'k')

    kde = logper_kde(rbin=rbin,width=width,raw=True,normed=True)
    kdenorm = kde(logps).max() * 3
    plt.plot(10**logps,kde(logps)/kdenorm,'k')
    plt.xscale('log')
    plt.xlim((0.5,MAXP+10))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))
    plt.xticks([1,3,10,30,100])
    plt.yticks([])
    plt.ylabel('Probability Density')
    #plt.ylabel('Signal-to-Noise Ratio')
    plt.xlabel('Orbital Period [days]')
    #plt.ylim(ymax=1.3*kde(logps).max()/kde)

    #plt.annotate('$SNR=%.1f$\n$P = %.2f$ d\n$R_p = %.2f R_e$\n$R_s = %.2f R_\odot$' %
    #             (props['SNR'],props['period'],props['Rp'],props['Rstar']),
    #             xy=(props['period'],props['SNR']),xytext=(50,8),
    #             textcoords='offset points',arrowprops=dict(arrowstyle="->",lw=2),
    #             fontsize=14,bbox=dict(boxstyle='round',fc='w'))

    plt.annotate('Smallest planets\n($R_p < %.1f R_e$)' % (rbin[1]),
                 xy=(1,kde(0)/kdenorm),ha='center',
                 xytext=(0.2,0.15),textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->',lw=2),
                 fontsize=16)
    plt.annotate('All Cool KOIs',
                 xy=(0.5,0.5),xycoords='axes fraction',
                 bbox=dict(boxstyle='round',fc='w',color='gray'),
                 alpha=0.5,
                 fontsize=16)

    kdeall = logper_kde(raw=True,normed=True,width=0.2)
    kdeallnorm = kdeall(logps).max() * 2
    #plt.plot(10**logps,kdeall(logps),'k',lw=3,zorder=1)
    plt.fill_between(10**logps,kdeall(logps)/kdeallnorm,logps*0,color='k',alpha=0.15)

    plt.ylim(ymax=plt.axis()[3]*1.2)
    threshs = []
    for k in kois:
        props = SNRprops(k)
        newsnrs = SNR_of_P(props,10**logps)
        xthresh = 10**logps[np.argmin(np.absolute(newsnrs-7.1))]
        threshs.append(xthresh)
        plt.axvline(xthresh,ls='--',color='r')
        #plt.annotate(k,xy=(xthresh*0.85,plt.axis()[3]*0.8),rotation=90,
        #             color='r',va='center',fontsize=14)
        #plt.annotate('(%.1fd, %.2f $R_e$)' % (props['period'],props['Rp']),
        #             xy=(xthresh*1.05,plt.axis()[3]*0.8),rotation=90,color='r',va='center')

    #plt.annotate('Orbital Periods\nresulting in SNR = 7.1',
    #             xy=(10**((np.log10(threshs[0]) + np.log10(threshs[1]))/2.),plt.axis()[3]*0.91),
    #             ha='center',color='r')

    x,y = (xthresh,0.8*plt.axis()[3])
    plt.annotate('SNR = 7.1 for $R_p = 1 R_\oplus$, $R_\star = 0.5 R_\odot$',
                 xy=(x,y),xytext=(0.05,0.9),textcoords='axes fraction',
                 color='r',fontsize=16,
                 arrowprops=dict(arrowstyle='->',lw=2,color='r'))
    
    plt.savefig('pincomplete.png')
    plt.savefig('pincomplete.pdf')
    
def plot_perdist_rbins(fig=None,width=0.2,scale=1,hist=False,bins=np.arange(-0.5,2,0.25)):
    kde1 = logper_kde(rbin=(0.,1.),width=width)
    kde2 = logper_kde(rbin=(1.,1.5),width=width)
    kde3 = logper_kde(rbin=(1.5,2.),width=width)
    kde4 = logper_kde(rbin=(2.,3.),width=width)

    plu.setfig(fig)

    if hist:
        plt.hist(kde1.data,lw=2,histtype='step',bins=bins)
        plt.hist(kde2.data,lw=2,histtype='step',bins=bins)
        plt.hist(kde3.data,lw=2,histtype='step',bins=bins)
        plt.hist(kde4.data,lw=2,histtype='step',bins=bins)
    else:    
        kde1.plot(fig=0,scale=scale)
        kde2.plot(fig=0,scale=scale)
        kde3.plot(fig=0,scale=scale)
        kde4.plot(fig=0,scale=scale)

def Pbin_fractions_old(pbins=np.logspace(0,np.log10(MAXP),6),rbins=[0,1,1.5,2,4],
                   plot=True,fig=None,xpos='vals',log=True,ylabels=True,
                   legend_bbox=(0.05, 0.6,0.3,0.3),legendcols=1,ticksright=False,
                   dN=5,zorder=3):

    fracs = []
    tots = []
    for i in range(len(pbins)):
        ns = []
        if i==0:
            plo = 0
        else:
            plo = pbins[i]
        if i==len(pbins)-1:
            continue
        phi = pbins[i+1]
        pok = ((ALLPERIODS > plo) & (ALLPERIODS < phi))
        for j in range(len(rbins)):
            rlo = rbins[j]
            if j==len(rbins)-1:
                continue
            rhi = rbins[j+1]
            rok = ((ALLRPS > rlo) & (ALLRPS < rhi))
            ns.append((pok & rok).sum())
        ns = np.array(ns)
        fracs.append(ns/float(ns.sum()))
        tots.append(ns.sum())
    tots = np.array(tots)
    fracs = np.array(fracs)
    pbins = np.array(pbins)
    if plot:
        plu.setfig(fig)
        ax = plt.gca()
        if ticksright:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        if xpos == 'inds':
            xs = np.arange(len(pbins)-1)
            width = 0.6
        elif xpos == 'vals':
            if log:
                logpbins = np.log10(pbins)
                #logpbins[0] = np.log10(0.6)
                #xs = 10**((logpbins[:-1] + logpbins[1:])/2)
                xs = 10**logpbins[:-1]*1.1
                #xs = 10**(np.sqrt(logpbins[:-1] * lopbins[1:]))
                #width = xs[1:] - xs[:-1]
                #width = np.concatenate((width,np.array([width[-1]])))
                width = xs*0.9
                #width *= 0.8
                #print pbins
                #print xs
            else:
                xs = (pbins[:-1] + pbins[1:])/2.
                width = (pbins[1:] - pbins[:-1])*0.8
        colors = ['b','g','r','c']
        for j in np.arange(len(rbins)-1)[::-1]:
            plt.bar(xs,fracs.cumsum(axis=1)[:,j]*tots,width,color=colors[j],
                    label='%.1f-%.1f $R_e$' % (rbins[j],rbins[j+1]),lw=2,zorder=zorder)

        plabels = []
        for i in range(len(pbins)-1):
            plabels.append('%i-%i days' % (pbins[i],pbins[i+1]))
        if ylabels:
            plt.ylabel('N')
            plt.yticks(np.arange(dN,tots.max()+dN,dN))
        else:
            plt.yticks([])
        plt.ylim(ymax=tots.max()*1.2)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1],labels[::-1],bbox_to_anchor=legend_bbox,loc=3,
               ncol=legendcols, mode="expand", borderaxespad=0.)
        #plt.legend(loc='upper left')
        plt.xticks(xs+width/2., plabels )
        if log:
            plt.xscale('log')
            plt.xlim(xmin = 0.4,xmax=MAXP+10)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))
            plt.xticks([1,2,4,8,16,32,MAXP])
        else:
            plt.xlim(xmin=-0.5)

    return fracs,tots

def Pbin_fractions(rkde,pbins=np.logspace(0,np.log10(MAXP),7),rbins=[0,1,1.5,2,4],
                   plot=True,fig=None,xpos='vals',log=True,ylabels=True,
                   legend_bbox=(0.05, 0.6,0.3,0.3),legendcols=1,ticksright=False,
                   dN=5,zorder=3):
    allps = rkde.survey.detections['P']
    allrps = rkde.survey.detections['Rp']
    fracs = []
    tots = []
    for i in range(len(pbins)):
        ns = []
        if i==0:
            plo = 0
        else:
            plo = pbins[i]
        if i==len(pbins)-1:
            continue
        phi = pbins[i+1]
        pok = ((allps > plo) & (allps < phi))
        for j in range(len(rbins)):
            rlo = rbins[j]
            if j==len(rbins)-1:
                continue
            rhi = rbins[j+1]
            rok = ((allrps > rlo) & (allrps < rhi))
            ns.append((pok & rok).sum())
        ns = np.array(ns)
        fracs.append(ns/float(ns.sum()))
        tots.append(ns.sum())
    tots = np.array(tots)
    fracs = np.array(fracs)
    pbins = np.array(pbins)
    if plot:
        plu.setfig(fig)
        ax = plt.gca()
        if ticksright:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        if xpos == 'inds':
            xs = np.arange(len(pbins)-1)
            width = 0.6
        elif xpos == 'vals':
            if log:
                logpbins = np.log10(pbins)
                #logpbins[0] = np.log10(0.6)
                #xs = 10**((logpbins[:-1] + logpbins[1:])/2)
                xs = 10**logpbins[:-1]*1.1
                #xs = 10**(np.sqrt(logpbins[:-1] * lopbins[1:]))
                #width = xs[1:] - xs[:-1]
                #width = np.concatenate((width,np.array([width[-1]])))
                width = xs*0.9
                #width *= 0.8
                #print pbins
                #print xs
            else:
                xs = (pbins[:-1] + pbins[1:])/2.
                width = (pbins[1:] - pbins[:-1])*0.8
        colors = ['b','g','r','c']
        for j in np.arange(len(rbins)-1)[::-1]:
            plt.bar(xs,fracs.cumsum(axis=1)[:,j]*tots,width,color=colors[j],
                    label='%.1f-%.1f $R_e$' % (rbins[j],rbins[j+1]),lw=2,zorder=zorder)

        plabels = []
        for i in range(len(pbins)-1):
            plabels.append('%i-%i days' % (pbins[i],pbins[i+1]))
        if ylabels:
            plt.ylabel('N')
            plt.yticks(np.arange(dN,tots.max()+dN,dN))
        else:
            plt.yticks([])
        plt.ylim(ymax=tots.max()*1.2)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1],labels[::-1],bbox_to_anchor=legend_bbox,loc=3,
               ncol=legendcols, mode="expand", borderaxespad=0.)
        #plt.legend(loc='upper left')
        plt.xticks(xs+width/2., plabels )
        if log:
            plt.xscale('log')
            plt.xlim(xmin = 0.4,xmax=MAXP+10)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))
            plt.xticks([1,2,4,8,16,32,64,MAXP])
        else:
            plt.xlim(xmin=-0.5)

    return fracs,tots


def plot_pdist_compare(fig=None,rbins=[0,1,1.5,2,4]):
    for i in range(len(rbins)):
        lo = rbins[i]
        if i==len(rbins)-1:
            continue
        else:
            hi = rbins[i+1]
        w = np.where((self.rps >= lo) & (self.rps < hi))
        n = np.size(w)
        kde = gaussian_kde(np.log10(self.periods[w]))
        plt.plot(logps,kde(logps),color=colors[i])    

def pdist_plot(rkde,fig=None):
    fracs,tots = Pbin_fractions(rkde,fig=fig,legend_bbox=(0.8, 0.61,0.3,0.3))
    plt.ylim(ymax=45)
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_coords(1.08, 0.25)
    plt.ylabel('N observed')
    plt.xlim((0.2,MAXP+10))
    logps = np.linspace(-1,np.log10(MAXP),500)
    logps = np.concatenate((logps,np.array([np.log10(MAXP)+0.001])))
    kde = rkde.survey.logperdist
    plt.plot(10**logps,kde(logps)*52,'k',lw=3,zorder=1)
    plt.fill_between(10**logps,kde(logps)*52,logps*0,color='k',alpha=0.15)
    plt.xlabel('Period [days]')
    plt.annotate('Probability Density $\phi_P$',xy=(-0.05,0.7),xycoords='axes fraction',
                 rotation=90,fontsize=16)
    plt.annotate('Implied all-planet period distribution',
                 xy=(0.035,0.86),xycoords='axes fraction',fontsize=16)
    plt.annotate('(corrected for transit probability)',
                 xy=(0.055,0.80),xycoords='axes fraction',fontsize=14)

    y0 = tots[0]/plt.axis()[3]+0.02
    plt.annotate('Observed planets',xy=(0.02,y0),xycoords='axes fraction',
                 fontsize=14)
    ax.legend().set_visible(False)
    plt.annotate('$2.0 R_e < R_p < 4.0 R_e$',xy=(0.025,y0-0.05),xycoords='axes fraction',
                 color='c',fontsize=12)
    plt.annotate('$1.5 R_e < R_p < 2.0 R_e$',xy=(0.025,y0-0.10),xycoords='axes fraction',
                 color='r',fontsize=12)
    plt.annotate('$1.0 R_e < R_p < 1.5 R_e$',xy=(0.025,y0-0.15),xycoords='axes fraction',
                 color='g',fontsize=12)
    plt.annotate('$0.5 R_e < R_p < 1.0 R_e$',xy=(0.025,y0-0.20),xycoords='axes fraction',
                 color='b',fontsize=12)
    plt.savefig('perdist.png')
    plt.savefig('perdist.pdf')
    

def pdist_plot_old(fig=None):
    fracs,tots = Pbin_fractions(fig=fig,legend_bbox=(0.8, 0.61,0.3,0.3))
    plt.ylim(ymax=45)
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_coords(1.08, 0.25)
    plt.ylabel('N observed')
    plt.xlim((0.25,MAXP+10))
    logps = np.linspace(-1,np.log10(MAXP),500)
    logps = np.concatenate((logps,np.array([np.log10(MAXP)+0.001])))
    kde = logper_kde()
    plt.plot(10**logps,kde(logps)*45,'k',lw=3,zorder=1)
    plt.fill_between(10**logps,kde(logps)*45,logps*0,color='k',alpha=0.15)
    plt.xlabel('Period [days]')
    plt.annotate('Probability Density $\phi_P$',xy=(-0.05,0.7),xycoords='axes fraction',
                 rotation=90,fontsize=16)
    plt.annotate('Implied all-planet period distribution',
                 xy=(0.035,0.79),xycoords='axes fraction',fontsize=16)
    plt.annotate('(corrected for transit probability)',
                 xy=(0.055,0.74),xycoords='axes fraction',fontsize=14)

    y0 = tots[0]/plt.axis()[3]+0.02
    plt.annotate('Observed planets',xy=(0.02,y0),xycoords='axes fraction',
                 fontsize=14)
    ax.legend().set_visible(False)
    plt.annotate('$2.0 R_e < R_p < 4.0 R_e$',xy=(0.025,y0-0.05),xycoords='axes fraction',
                 color='c',fontsize=12)
    plt.annotate('$1.5 R_e < R_p < 2.0 R_e$',xy=(0.025,y0-0.12),xycoords='axes fraction',
                 color='r',fontsize=12)
    plt.annotate('$1.0 R_e < R_p < 1.5 R_e$',xy=(0.025,y0-0.19),xycoords='axes fraction',
                 color='g',fontsize=12)
    plt.annotate('$0.5 R_e < R_p < 1.0 R_e$',xy=(0.025,y0-0.26),xycoords='axes fraction',
                 color='b',fontsize=12)
    plt.savefig('perdist.png')
    plt.savefig('perdist.pdf')
    
def plot_snrfns(fig=None,kois=['KOI961.01','KOI952.03'],maxq=MAXQ,xlabels=(6,30),
                xytexts=((0.2,0.75),(0.6,0.47)),folder='koi_snrfns_batalha',
                offsets=(0.17,-0.05)):
    plu.setfig(fig)
    for i,k in enumerate(kois):
        snrfn = koi_SNRfn(k,maxq=maxq,folder=folder)
        eta = snrfn.integrate_efficiency()
        snrfn.plot(fig=0,color='k',lw=3)
        x = xlabels[i]
        y = snrfn(x)
        props = SNRprops(k)
        plt.annotate('%s: $\eta_{disc} = %.2f$' % (k,eta),xy=(x,y),xytext=xytexts[i],
                     textcoords='axes fraction',
                     arrowprops=dict(arrowstyle="->",lw=2),fontsize=16)
        plt.annotate('$SNR=%.1f$\n$P = %.2f$ d\n$R_p = %.2f R_e$\n$R_s = %.2f R_\odot$' %
                     (props['SNR'],props['period'],props['Rp'],props['Rstar']),
                     xy=(xytexts[i][0]+offsets[0],xytexts[i][1]+offsets[1]),
                     xycoords='axes fraction',fontsize=14,bbox=dict(boxstyle='round',fc='w'),
                     va='top')
        plt.ylabel('Probability Density $\phi_{SNR}$')
    plt.savefig('snrexamples.png')
    plt.savefig('snrexamples.pdf')

def plot_radfn(rkde,fig=None):
    rh = rkde.plot(uncs=True,hist=True,histbins=np.arange(0.5,4.1,0.5),histymax=1.0,rmin=0.4,rmax=4,etalabel=False,fig=fig,return_histax=True)
    rkde.plot_hist(ls=':',fig=0,color='b',ax=rh)
    plt.savefig('rdist.png')
    plt.savefig('rdist.pdf')


def plot_radfn_old(rkde,fig=None,maxval=4,minval=0.4,histymax=0.75,histcolor='b',onlythis=False,
               histbins=None,widthfactor=1,plawnorm=4,plotplaw=True):
    plu.setfig(fig)
    host = host_subplot(111)#, axes_class=AA.Axes)

    #rkde = KOI_radiusKDE(tablefile='rkde.txt',maxval=maxval,widthfactor=widthfactor)
    #rkde.bootstrap(1000)
    

    rh,vals = rkde.plot(fig=0,hist=True,host=host,uncs=True,histymax=histymax,
                     histcolor=histcolor,return_histvals=True,histbins=histbins,
                     label='(1) This work (SNR ramp + period correction)\n     (%.1f planets/star)' % quad(rkde,minval,maxval)[0])
    #rkde.weights = 1./rkde.transprobs
    #rkde._setfns()
    #rkde.plot(fig=0,ls=':')
    if not onlythis:
        rkde = KOI_radiusKDE('rkdesimple.txt',maxval=maxval,widthfactor=widthfactor)
        rkde.plot(fig=0,ls='--',lines=False,hist=True,host=host,
                  histymax=histymax,histlabel=False,histcolor=histcolor,histbins=histbins,
                  label='(2) SNR ramp, no period correction\n     (%.1f planets/star)' % quad(rkde,minval,maxval)[0])

        rkde = KOI_radiusKDE(tablefile='rkdesimplethresh.txt',maxval=maxval,widthfactor=widthfactor)
        rkde.plot(fig=0,ls=':',lines=False,hist=True,host=host,
                  histymax=histymax,histlabel=False,histcolor=histcolor,histbins=histbins,
                  label='(3) 7.1 threshold, no period correction\n     (%.1f planets/star)' % quad(rkde,minval,maxval)[0])

        rh.annotate('Occurrence rates in bins used by\nDressing & Charbonneau (2013)',
                    xy=(2.4,vals[4]),xycoords='data',color=histcolor,
                    xytext=(-5,30),textcoords='offset points',
                    arrowprops=dict(arrowstyle='->',lw=2,color=histcolor),
                    fontsize=12)

    plt.xlim((minval,maxval))


    ymax = plt.axis()[3]
    plaw = utils.powerlaw(-2,0.5,4,norm=plawnorm)
    rps = np.arange(0.9,4,0.02)
    if plotplaw:
        if onlythis:
            plt.plot(rps,plaw(rps),'g:',lw=3)
        else:
            plt.plot(rps,plaw(rps),'g',lw=3)
        plt.annotate('$R^{-2}$',xy=(0.92*maxval,1.1*plaw(0.92*maxval)),color='g',fontsize=14)

    plt.ylim(ymax=ymax)

    plt.annotate('Inverse detection efficiencies',
                 xy=(0.55,0.15),bbox=dict(boxstyle='square',fc='w'),
                 fontsize=12,color='r',va='center')



    if onlythis:
        #pass
        plt.annotate('%.1f planets per cool star\n       (P < %i days)' %\
                     (quad(rkde,minval,maxval)[0],MAXP),fontsize=16,
                     xy=(0.53,0.85),xycoords='axes fraction')
    else:
        plt.legend(prop=dict(size=12))
    #plt.savefig('rdist.png')
    #plt.savefig('rdist.pdf')

def plot_rdist(widthfactor=1,plawnorm=3.5,**kwargs):
    plot_radfn(histbins=np.arange(0,4.1,0.5),histymax=0.9,widthfactor=widthfactor,
               plawnorm=plawnorm,onlythis=True,**kwargs)
    plt.savefig('rdist.png')
    plt.savefig('rdist.pdf')

def plot_rcompare_old(**kwargs):
    plot_radfn(widthfactor=1.5,plotplaw=False,plawnorm=2,**kwargs)
    plt.savefig('rcompare.png')
    plt.savefig('rcompare.pdf')    

def plot_mratdist(fig=None,width=None,ssystem=False,log=True,maxval=-3.8,minval=-6):
    """default width is 0.05
    """
    if log:
        mkde = KOI_massratKDE('mkde.txt',minval=minval)
    else:
        mkde = KOI_massratKDE('mkdelin.txt',minval=minval)
        maxval = 10**maxval
    if width is not None:
        mkde.set_width(width)
    mkde.plot(lines=True,maxval=maxval,minval=minval,color='k')
    ymax = plt.axis()[3]
    if ssystem:
        for k in JOVIANMASSRATS:
            massrat = JOVIANMASSRATS[k]
            plt.annotate(k,xy=(massrat*1.02,ymax*0.9),rotation=90,va='center')
            plt.axvline(massrat,ls=':',color='k')
        for k in SSMASSRATS:
            massrat = SSMASSRATS[k]
            plt.annotate(k,xy=(massrat*1.02,ymax*0.9),rotation=90,va='center',color='b')
            plt.axvline(massrat,ls=':',color='b')

    plt.annotate('Cool KOI planets',xy=(0.4,0.32),xycoords='axes fraction',
                 ha='center',color='r',fontsize=14)

    plt.savefig('massrat.png')
    plt.savefig('massrat.pdf')


def plot_PR(fig=None,zoomed=True,ms=15,scaledsize=True):
    plu.setfig(fig)

    if USE_Q1Q8:
        kois = KOIS_Q1Q8
    else:
        kois = KOIS_Q1Q12

    for k in ALLKOIS:
        if scaledsize:
            s = ms * kois[k]['Rstar']/0.5
        plt.semilogx(kois[k]['P'],kois[k]['rp'],'ko',ms=s,mfc='none')

    plt.xlim(xmin=0.4)
    if zoomed:
        plt.ylim(ymax=5)
    else:
        plt.ylim(ymax=12)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))
    plt.xlabel('Period [days]')
    plt.ylabel('Planet Radius [Re]')

def plot_simplehist(fig=None,binsize=0.3,**kwargs):
    plu.setfig(fig)
    bins = np.arange(0,4,binsize)
    plt.hist(ALLRPS,bins=bins,histtype='step',color='k',lw=3,**kwargs)
    


def plot_snrdist(rkde,fig=None,etafn_color='b',plot_inset=False):
    plu.setfig(fig)
    
    h1,bins1 = np.histogram(rkde.survey.SNRdist_samples,bins=np.arange(0,100,0.3),normed=True)
    h2,bins2 = np.histogram(rkde.survey.SNRdist_samples*4,bins=np.arange(100),normed=True)

    fn1 = interpolate((bins1[1:] + bins1[:-1])/2.,h1,s=0)
    fn2 = interpolate((bins2[1:] + bins2[:-1])/2.,h2,s=0)

    host = host_subplot(111)

    plt.plot(bins1,fn1(bins1),'k',lw=2)
    plt.plot(bins2,fn2(bins2),'k',lw=2)

    plt.ylim(ymin=0)
    plt.ylabel('Probability Density $\phi_{SNR}$',fontsize=18)
    plt.yticks([])
    plt.xlabel('SNR',fontsize=18)

    textcolor = 'k'
    if plot_inset:
        textcolor='r'

    plt.annotate('$R_p = 1.0 R_\oplus$, $\eta_{disc} = %.2f$' % rkde.survey.etadisc_of_r(1.),
                 xy=(2.5,fn1(2.5)),xycoords='data',fontsize=16,color=textcolor,
                 xytext=(0.1,0.8),textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->',lw=2,color=textcolor))
                 #bbox=dict(boxstyle='round',fc='w'))

    plt.annotate('$R_p = 2.0 R_\oplus$, $\eta_{disc} = %.2f$' % rkde.survey.etadisc_of_r(2.),
                 xy=(35,fn2(35)),xycoords='data',fontsize=16,color=textcolor,
                 xytext=(0.6,0.3),textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->',lw=2,color=textcolor))
                 #bbox=dict(boxstyle='round',fc='w'))


    rh = host.twinx()
    rh.axis['right'].toggle(all=True)

    rh.plot(bins1,rkde.survey.etafn(bins1),color=etafn_color,ls=':')
    rh.yaxis.label.set_color(etafn_color)
    rh.set_ylim(ymax=1.3)
    rh.set_yticks(np.arange(0,1.1,0.2))
    rh.tick_params(axis='y', colors=etafn_color)
    rh.spines['right'].set_color(etafn_color)
    rh.set_ylabel('Signal detection efficiency $\eta(SNR)$',labelpad=10)

    plt.xlim(xmax=50)

    if plot_inset:
        ax = plt.axes([0.55,0.57,0.3,0.3])
        rps = np.arange(0,5,0.1)
        ax.plot(rps,rkde.survey.etadisc_of_r(rps),'r')
        ax.set_yticks([0,0.5,1.0])
        ax.set_xticks(np.arange(0,5.1,1))
        ax.set_xlabel('$R_p$ $[R_\oplus]$',labelpad=8,color='r')
        ax.set_ylabel('$\eta_{disc}$',color='r',labelpad=-2)

    fig = plt.gcf()
    #fig.subplots_adjust(bottom=0.1,right=0.9,left=0.1)

def plot_etadisc_of_r(rkde,fig=None):
    plu.setfig(fig)
    rps = np.arange(0,4.1,0.1)

    plt.plot(rps,rkde.survey.etadisc_of_r(rps),'k')
    plt.xlabel('Planet Radius [Earth Radii]',labelpad=5,fontsize=18)
    plt.ylabel('Discovery efficiency $\eta_{disc}$',fontsize=18)
    

def plot_rdist_compare(rkde,fig=None):
    rkde.set_normal()
    rkde.plot(fig=fig,uncs=True,lines=False,npps_label=False,
              label='Period correction, survey- and \nperiod-averaged transit\nprobability, de-biased (this work)')

    rkde.set_simple(transprob_simple=True)
    rkde.plot(fig=0,uncs=False,lines=True,npps_label=False,ls='--',
              label='No period correction,\nindividual transit probabilities\nno de-biasing')
    
    rkde.set_simple_thresh(transprob_simple=True)
    rkde.plot(fig=0,uncs=False,lines=False,npps_label=False,ls=':',
              label='No period correction,\nindividual transit probabilities,\nSNR thresh, no de-biasing')

    rkde.set_normal()

    plt.legend(bbox_to_anchor=(1.1,0.95))
    

def plot_rdist_transprob_compare(rkde,fig=None,third=True,second=False):
    rkde.set_normal()
    npps1 = quad(rkde,0,4)[0]
    rkde.plot(fig=fig,uncs=True,lines=False,npps_label=False,
              label='Survey- and period-averaged\ntransit probability,\nand period correction (this work)')
    if second:
        rkde.set_transprob_simple()
        npps2 = quad(rkde,0,4)[0]
        rkde.plot(fig=0,uncs=False,lines=False,npps_label=False,ls=':',
                  label='Individual transit probabilities')
    if third:
        rkde.set_simple(transprob_simple=True)
        npps3 = quad(rkde,0,4)[0]
        rkde.plot(fig=0,uncs=False,lines=True,npps_label=False,ls='--',
                  label='Individual transit probabilities,\nno period correction')
        
    rkde.set_normal()


    plt.legend()

def plot_rdist_compare_old(rkde,fig=None):
    rkde.set_normal()
    npps1 = quad(rkde,0,8)[0]
    rkde.plot(fig=fig,uncs=True,lines=False,npps_label=False,
              label='This work (%.1f planets/star)' % npps1) 
    rkde.set_simple(transprob_simple=False)
    npps2 = quad(rkde,0,8)[0]
    rkde.plot(fig=0,uncs=False,lines=True,npps_label=False,ls='--',
              label='No period correction (%.1f planets/star)' % npps2)
    rkde.set_simple_thresh(transprob_simple=False)
    npps3 = quad(rkde,0,8)[0]
    rkde.plot(fig=0,uncs=False,lines=False,npps_label=False,ls=':',
              label='No period correction, SNR threshold\n (%.1f planets/star)' % npps3)
    rkde.set_normal()
    
    plt.legend()

def plot_eta_compare(rkde,fig=None,labelinds=False,factor=3):
    plu.setfig(fig)
    d_etas = rkde.etadiscs - rkde.survey.etadisc_simple
    #d_etas = rkde.etadiscs
    inds = np.argsort(rkde.survey.detections['Rp'][::-1])
    for i in inds:
        post = rkde.posteriors[i]
        outer = post.pctile(0.84)*factor
        inner = post.pctile(0.16)*factor
        middle = (outer + inner)/2
        width = outer - inner
        #plt.semilogx(rkde.survey.detections['P'][i],d_etas[i],
        #         marker='o',ms=inner,mew=width,mec='grey',
        #         color='k',mfc='none')
        P = rkde.survey.detections['P'][i]
        plt.semilogx(rkde.survey.detections['P'][i],d_etas[i],
                     marker='o',ms=outer,
                     color='k',mfc='k',alpha=0.2,mew=0)
        plt.semilogx(rkde.survey.detections['P'][i],d_etas[i],
                     marker='o',ms=inner,
                     color='k',mfc='w',mew=0.5)
        plt.semilogx(rkde.survey.detections['P'][i],d_etas[i],
                     marker='o',ms=outer,
                     color='k',mfc='none',mew=0.5)
        #plt.semilogx([P,P],[d_etas[i],rkde.survey.etadisc_simple[i]],'k',lw=1)
        if labelinds:
            plt.annotate(i,xy=(np.log10(rkde.survey.detections['P'][i]),d_etas[i]),
                         xycoords='data',fontsize=10)
    ax = plt.gca()
    plt.xlabel('Period [days]',labelpad=0)
    plt.ylabel('$\eta_{disc,i} - \eta_{disc,i}^{simple}$',fontsize=18,labelpad=-5)
    #plt.ylabel('$\Delta \eta_{disc,i}$',fontsize=18,labelpad=-5)

    plt.semilogx(0.6,0.3,'k',marker='o',ms=2*factor,mew=0.5,mfc='none')
    plt.annotate('$2 R_\oplus$',xy=(0.7,0.3),xycoords='data',va='center',fontsize=20)

    Ps = np.logspace(-1,2.5,10)
    plt.fill_between(Ps,-0.5*np.ones(10),np.zeros(10),color='g',alpha=0.1,zorder=5)
    plt.fill_between(Ps,np.zeros(10),0.5*np.ones(10),color='r',alpha=0.1,zorder=5)

    plt.annotate('Simple method\noverestimates completeness',xy=(3,-0.4),xycoords='data',color='g',va='center',fontsize=16)
    plt.annotate('Simple method\nunderestimates completeness',xy=(3,0.35),xycoords='data',color='r',va='center',fontsize=16)


    plt.xlim((0.3,200))
    plt.ylim((-0.5,0.5))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))

    plt.xticks([1,3,10,30,100])

    plt.draw()

def plot_rdist_compare_pdist(rkde,rkde2,fig=None,xmin=0.3):
    plu.setfig(fig)
    rkde.plot(uncs=True,npps_label=False,lines=True,label='Observed\nperiod distribution')
    rkde2.plot(uncs=False,npps_label=False,lines=False,ls='--',fig=0,label='Logflat + exponential\nperiod distribution')

    plt.legend(loc='lower right',bbox_to_anchor=(1.0,0.16))
    plt.xlim(xmin=xmin)

    ax = plt.axes([0.5,0.57,0.35,0.3])
    pers = np.linspace(0.5,MAXP,500)
    ax.semilogx(pers,rkde.survey.logperdist(np.log10(pers)),'k')
    ax.semilogx(pers,rkde2.survey.logperdist(np.log10(pers)),'k--')
    ax.set_yticks([])
    ax.set_xlim((0.5,MAXP))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))
    ax.set_xticks([1,3,10,30,100])
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('$\phi_P$')

    

def fit_broken_plaw(rkde,p0=(-1,-3,1.2),rmin=0.75,rmax=4.0,errors=True):
    #if not hasattr(rkde,'uncfn_p'):
    if rkde.uncfn_p is None:
        rkde.bootstrap()
    
    norm = quad(rkde,rmin,rmax)[0]
    rgrid = np.arange(rmin,rmax,0.01)
    errbar = (rkde.uncfn_p(rgrid) + rkde.uncfn_m(rgrid))/2.
    rfn = rkde(rgrid)
    def objfn(pars):
        broken_plaw = utils.broken_powerlaw(*pars,xmin=rmin,xmax=rmax,norm=norm)
        resids = ((broken_plaw(rgrid) - rfn))
        if errors:
            resids /= errbar
        return resids

    return utils.broken_powerlaw(*leastsq(objfn,p0)[0],xmin=rmin,xmax=rmax,norm=norm)

def fit_triple_plaw(rkde,p0=(2,-1,-3,0.7,1.3),rmin=0.5,rmax=4.0,errors=True):
    #if not hasattr(rkde,'uncfn_p'):
    if rkde.uncfn_p is None:
        rkde.bootstrap()
    
    norm = quad(rkde,rmin,rmax)[0]
    rgrid = np.arange(rmin,rmax,0.01)
    errbar = (rkde.uncfn_p(rgrid) + rkde.uncfn_m(rgrid))/2.
    rfn = rkde(rgrid)
    def objfn(pars):
        triple_plaw = utils.triple_powerlaw(*pars,xmin=rmin,xmax=rmax,norm=norm)
        resids = ((triple_plaw(rgrid) - rfn))
        if errors:
            resids /= errbar
        return resids

    return utils.triple_powerlaw(*leastsq(objfn,p0)[0],xmin=rmin,xmax=rmax,norm=norm)


def fit_broken_plaw(rkde,p0=(-1,-3,1.2),rmin=0.75,rmax=4.0,errors=True):
    #if not hasattr(rkde,'uncfn_p'):
    if rkde.uncfn_p is None:
        rkde.bootstrap()
    
    norm = quad(rkde,rmin,rmax)[0]
    rgrid = np.arange(rmin,rmax,0.01)
    errbar = (rkde.uncfn_p(rgrid) + rkde.uncfn_m(rgrid))/2.
    rfn = rkde(rgrid)
    def objfn(pars):
        broken_plaw = utils.broken_powerlaw(*pars,xmin=rmin,xmax=rmax,norm=norm)
        resids = ((broken_plaw(rgrid) - rfn))
        if errors:
            resids /= errbar
        return resids

    return utils.broken_powerlaw(*leastsq(objfn,p0)[0],xmin=rmin,xmax=rmax,norm=norm)


SIMSURVEY_SNRDIST = Table.read('simsurvey_snrdist.fits')
SIM_RS, SIM_ES = np.loadtxt('simsurvey_etadisc.txt',unpack=True)

class SimCoolKOISurvey(occ.TransitSurvey):
    def __init__(self,maxq=MAXQ,recalc=False,rdist=None,logperdist=None,NPPS=1.,
                 etadisc=None,etafn=occ.etaPT,maxp=MAXP,basename='simsurvey',
                 simple_transitdepth=False,targets=None,snrdist=SIMSURVEY_SNRDIST,
                 barebones=False,etafn_es=SIM_ES,etafn_rs=SIM_RS,**kwargs):
        if rdist is None:
            #rdist = utils.broken_powerlaw(2,-3,0.8,xmin=0.5,xmax=4.0,norm=NPPS)
            #rdist = utils.triple_powerlaw(4.1878,-1.5447,-5.8474,0.80887,2.1975,xmin=0.5,xmax=4.0,
            #                              norm=NPPS)
            rdist = utils.triple_powerlaw(2,0,-2,0.75,1.2,xmin=0.5,xmax=4.0,
                                          norm=NPPS)            
        if logperdist is None:
            logperdist = occ.ToyLogPdist(10,0.5,0.5,150)


        self.logperdist = logperdist
        self.rdist = rdist

        if targets is None:
            targets = Table.read('coolkois_q1q12_targets.txt',format='ascii')
        #targets = pd.read_hdf('coolkois_q1q12_targets.h5','table')

        N = int(len(targets) * rdist.norm)

        pers = 10**logperdist.resample(N)
        try:
            rps = rdist.resample(N)
        except AttributeError:
            rps = rdist.draw(N)

        inds = rand.randint(len(targets),size=N)

        semimajors = np.array(occ.semimajor(pers,targets['M'][inds]))
        incs = np.arccos(rand.random(N))
        bs = np.array(semimajors*AU*np.cos(incs) / (targets['R'][inds]*RSUN))
        
        SNRs = np.array(bs*0)

        tra = (bs < 1)
        SNRs[np.where(~tra)] = 0
        wtra = np.where(tra)

        SNRs[wtra] = occ.transitSNR(pers[wtra],rps[wtra],np.array(targets['R'])[inds][wtra],
                                    np.array(targets['M'])[inds][wtra],bs[wtra],
                                    np.array(targets['Tobs'])[inds][wtra],
                                    np.array(targets['noise'])[inds][wtra],
                                    np.array(targets['u1'])[inds][wtra],
                                    np.array(targets['u2'])[inds][wtra],npts=50,noise_timescale=3,
                                    force_1d=True,simple=simple_transitdepth)
        
        pr_detect = etafn(SNRs)
        u = rand.random(size=N)
        detected = u < pr_detect
        w_detected = np.where(detected)
        
        names = np.arange(detected.sum())+1

        detections = Table(data=[names,np.array(targets['ID'])[inds][w_detected],
                                 rps[w_detected],0.2*rps[w_detected],
                                 pers[w_detected],SNRs[w_detected]],
                           names=['name','ID','Rp','dRp','P','SNR'])

        if len(detections)==0:
            raise ZeroDetectionsError

        self.allplanets = Table(data=[rps,pers,SNRs,bs,u],
                                names=['Rp','P','SNR','b','u'])

        if barebones:
            print 'barebones not implemented yet.'

        occ.TransitSurvey.__init__(self,targets,detections,logperdist=logperdist,
                                   etadisc_filename='%s_etadisc.txt' % basename,
                                   snrdist_filename='%s_snrdist.fits' % basename,
                                   snrdist=snrdist,etafn_rs=etafn_rs,
                                   etafn_es=etafn_es,
                                   recalc=recalc,maxp=maxp)
                                   
class SimRKDE(occ.RadiusKDE_FromSurvey):
    def __init__(self,survey=None,maxp=MAXP,rmax=4,**kwargs):
        if survey is None:
            survey = SimCoolKOISurvey(**kwargs)

        occ.RadiusKDE_FromSurvey.__init__(self,survey,maxval=rmax,maxp=maxp,usefpp=False,
                                          truedist=survey.rdist)


def manysims(folder='simtests',N=100,rdist=None,bootstrap=True):

    for i in np.arange(N):
        print '%i of %i' % (i+1,N)
        file1 = '%s/wkde%i.h5' % (folder,i)
        if os.path.exists(file1):
            os.remove(file1)
        simkde = SimRKDE(rdist=rdist)
        if bootstrap:
            simkde.bootstrap()
        simkde.save(file1)

        file2 = '%s/wkde_simple%i.h5' % (folder,i)
        if os.path.exists(file2):
            os.remove(file2)
        simkde.set_simple()
        if bootstrap:
            simkde.bootstrap(N=200)
        simkde.save(file2)


def plot_methodcompare(folder='simtests2',fig=None,**kwargs):
    plu.setfig(fig)
    plotsims(folder=folder,fig=0,color='r',
             label='Period redistribution,\naverage transit probability',
             **kwargs)
    plotsims(folder=folder,fig=0,color='b',simple=True,alpha=0.3,
             label='No period redistribution,\nindividual transit probabilities',
             **kwargs)

    plt.ylim(ymax=4)
    plt.legend()
    #plt.yticks(np.arange(0,4,0.5))
    plt.xlabel('Planet Radius [Earth]')
    plt.ylabel('Radius distribution function $\phi_r$')

def plot_simhists(folder='simtests2',fig=None,simple=False,
                  N=100,bins=np.logspace(np.log10(0.5),np.log10(4),7),
                  ymax=1.5,**kwargs):
    plu.setfig(fig)
    for i in np.arange(N):
        if simple:
            filename = '%s/wkde_simple%i.h5' % (folder,i)
        else:
            filename = '%s/wkde%i.h5' % (folder,i)
        try:
            w = occ.RadiusKDE_FromH5(filename,norm=3893)
            w.plot_hist(fig=0,lw=0.5,bins=bins,**kwargs)

        except IOError:
            pass
        except KeyError:
            pass
    plt.ylim(ymax=ymax)
    
def calc_npps_sims(folder='simtests2',simple=False,N=100,rmin=0.5):

    npps_debiased = []
    npps_raw = []
    for i in np.arange(N):
        if simple:
            filename = '%s/wkde_simple%i.h5' % (folder,i)
        else:
            filename = '%s/wkde%i.h5' % (folder,i)
        try:
            w = occ.RadiusKDE_FromH5(filename,norm=3893)
            fn = lambda x: w(x) + w.bias(x)
            npps_debiased.append(quad(w,rmin,4,full_output=1)[0])
            npps_raw.append(quad(fn,rmin,4,full_output=1)[0])

        except IOError:
            pass
        except KeyError:
            pass    
    return (quad(w.truedist,rmin,4,full_output=1)[0],
            np.array(npps_debiased),np.array(npps_raw))

def plot_bias_all(fig=None,ms=10):
    plot_npps_bias(fig=fig,color='k',plot_color='k',ms=ms)
    plot_npps_bias(fig=0,rmin=1.0,color='g',plot_color='g',ms=ms)
    plot_npps_bias(fig=0,rmin=1.5,color='b',plot_color='b',ms=ms)
    plot_npps_bias(fig=0,rmin=2.0,color='r',plot_color='r',ms=ms)

    plt.ylim(ymax=2.6)
    plt.xticks([1,2],['With\nPeriod redistribution',
                      'Without\nPeriod redistribution'])

    l1 = plt.Line2D([1],[1],marker='s',color='k',mfc='w',ls='none')
    l2 = plt.Line2D([1],[1],marker='o',color='k',mfc='w',ls='none')
    plt.legend([l2,l1],['de-biased','not de-biased'],loc='lower center',numpoints=1)


def plot_npps_bias(folder='simtests2',fig=None,rmin=0.5,ax=None,
                   color='r',ms=10,plot_color='k'):
    if ax is None:
        plu.setfig(fig)
    npps,npps_debiased,npps_raw = calc_npps_sims(folder=folder,rmin=rmin)
    npps,npps_debiased_simple,npps_raw_simple = calc_npps_sims(folder=folder,
                                                               simple=True,
                                                               rmin=rmin)

    if ax is None:
        ax = plt.gca()

    ax.errorbar(0.9,npps_debiased.mean(),yerr=npps_debiased.std(),
                color=plot_color,marker='o',mfc='w',ms=ms,mec=plot_color)
    ax.errorbar(1.1,npps_raw.mean(),yerr=npps_debiased.std(),
                color=plot_color,marker='s',mfc='w',ms=ms,mec=plot_color)
    ax.errorbar(1.9,npps_debiased_simple.mean(),yerr=npps_debiased.std(),
                color=plot_color,marker='o',mfc='w',ms=ms,mec=plot_color)
    ax.errorbar(2.1,npps_raw_simple.mean(),yerr=npps_debiased.std(),
                color=plot_color,marker='s',mfc='w',ms=ms,mec=plot_color)

    ax.axhline(npps,color=color,ls=':')
    ax.set_xlim((0.5,2.5))

    ax.annotate('$r > %.1f R_\oplus$' % rmin,xy=(2.47,npps),xycoords='data',
                fontsize=18,color=color,ha='right')

    ax.set_ylabel('Number of planets per star')



    plt.draw()

def plotsims(folder='simtests2',fig=None,simple=False,N=100,
             uncs=True,npts=500,plot_all=True,color='r',ls='-',alpha=0.5,
             label=None,xmin=0.2,xmax=4.,hists=False,
             histbins=np.logspace(np.log10(0.5),np.log10(4),7),
             no_bias_correct=False):
    plu.setfig(fig)
    xs = np.linspace(xmin,xmax,npts)
    tots = np.zeros((N,npts))
    histvals = np.zeros((N,len(histbins)-1))
    for i in np.arange(N):
        if simple:
            filename = '%s/wkde_simple%i.h5' % (folder,i)
        else:
            filename = '%s/wkde%i.h5' % (folder,i)
        try:
            w = occ.RadiusKDE_FromH5(filename,norm=3893)
            if plot_all:
                if no_bias_correct:
                    plt.plot(xs,w(xs) + w.bias(xs),color=color,lw=1,zorder=1,
                             label=None,alpha=0.2)
                else:
                    w.plot(lines=False,fig=0,color=color,alpha=0.2,lw=1,zorder=1,
                           label=None,minval=xmin)
            if hists:
                histvals[i,:] = w.plot_hist(noplot=True,bins=histbins)
            if no_bias_correct:
                tots[i,:] = w(xs) + w.bias(xs)
            else:
                tots[i,:] = w(xs)
            #print len(w.data)

        except IOError:
            tots = np.delete(tots,np.s_[i:],0)
            pass
        except KeyError:
            tots = np.delete(tots,np.s_[i:],0)
            pass
    sorted = np.sort(tots,axis=0)
    pvals = sorted[-N*16/100,:] #84th pctile pts
    medvals = sorted[N/2,:]  #50th pctile pts
    mvals = sorted[N*16/100,:] #16th pctile pts

    histmeds = np.median(histvals,axis=0)
    if hists:
        for i in np.arange(len(histbins)-1)+1:
            lo = histbins[i-1]
            hi = histbins[i]
            val = histmeds[i-1]
            xs = [lo,hi]
            ys = [val,val]
            plt.plot(xs,ys,color=color,lw=3)
    
    plt.plot(xs,medvals,color='w',lw=5,zorder=2,label=None)
    plt.plot(xs,medvals,color=color,ls=ls,lw=3,zorder=2,label=label)
    plt.fill_between(xs,pvals,mvals,color=color,alpha=alpha,zorder=2,label=None)
    plt.plot(xs,w.truedist(xs),'k--',lw=5,zorder=2,label=None)
    plt.plot(xs,w.truedist(xs),'w--',lw=3,zorder=2,label=None)


def sim_ensemble(amin=-3,amax=1,da=1,N=100,
                 norms=[0.5,1,2,4],rmin=0.25,rmax=4):
    alphas = np.arange(amin,amax+da,da)

    alphalist = []
    normlist = []
    nppslist_05 = []
    nppslist_10 = []
    nppslist_15 = []
    nppslist_20 = []
    truenpps_05 = []
    truenpps_10 = []
    truenpps_15 = []
    truenpps_20 = []
    nlist = []

    nlist_05 = []
    nlist_10 = []
    nlist_15 = []
    nlist_20 = []

    widgets = ['Doing simulations...',Percentage(),' ',
               Bar(marker=RotatingMarker()),' ',ETA()]
    pbar = ProgressBar(widgets=widgets,maxval=len(alphas)*len(norms)*N)
    pbar.start()
    
    targets = Table.read('coolkois_q1q12_targets.txt',format='ascii')    

    i=0
    for a in alphas:
        for norm in norms:
            for j in range(N):
                i += 1
                rdist = utils.powerlaw(a,rmin,rmax,norm=norm)
                try:
                    simkde = SimRKDE(rdist=rdist,targets=targets)
                except ZeroDetectionsError:
                    continue
                alphalist.append(a)
                normlist.append(norm)
            
                truenpps_05.append(quad(rdist,0.5,rmax,full_output=1)[0])
                truenpps_10.append(quad(rdist,1.0,rmax,full_output=1)[0])
                truenpps_15.append(quad(rdist,1.5,rmax,full_output=1)[0])
                truenpps_20.append(quad(rdist,2.0,rmax,full_output=1)[0])
                
                w05 = np.where(simkde.survey.detections['Rp'] > 0.5)
                w10 = np.where(simkde.survey.detections['Rp'] > 1.0)
                w15 = np.where(simkde.survey.detections['Rp'] > 1.5)
                w20 = np.where(simkde.survey.detections['Rp'] > 2.0)
                nppslist_05.append(np.sum(simkde.weights_simple[w05])/simkde.norm)
                nppslist_10.append(np.sum(simkde.weights_simple[w10])/simkde.norm)
                nppslist_15.append(np.sum(simkde.weights_simple[w15])/simkde.norm)
                nppslist_20.append(np.sum(simkde.weights_simple[w20])/simkde.norm)
                
                nlist_05.append(len(simkde.weights_simple[w05]))
                nlist_10.append(len(simkde.weights_simple[w10]))
                nlist_15.append(len(simkde.weights_simple[w15]))
                nlist_20.append(len(simkde.weights_simple[w20]))
                nlist.append(simkde.N)
                pbar.update(i)
    pbar.finish()
            
    d = dict(alpha=alphalist,norm=normlist,n=nlist,
             n_05=nlist_05,n_10=nlist_10,
             n_15=nlist_15,n_20=nlist_20,
             npps_05=nppslist_05,npps_10=nppslist_10,
             npps_15=nppslist_15,npps_20=nppslist_20,
             truenpps_05=truenpps_05,truenpps_10=truenpps_10,             
             truenpps_15=truenpps_15,truenpps_20=truenpps_20)

    return pd.DataFrame(d)

def simresults_df(folder='npps_sims'):
    files = glob.glob(folder+'/*.h5')
    df = pd.DataFrame()
    for f in files:
        df = df.append(pd.read_hdf(f,'table'))
    return df

def summarize_simresults(df=None,filename='npps_simresults.h5',fig=None,
                         plot=True,folder='npps_sims'):
    if folder is not None:
        df = simresults_df(folder)

    if df is None:
        df = pd.read_hdf(filename,'table')
    
    if plot:
        plu.setfig(fig)

    alphas = df['alpha'].unique()
    norms = df['norm'].unique()
    
    k=0
    for i,a in enumerate(alphas):
        for j,n in enumerate(norms):
            print 'alpha = %.2f, norm = %.2f' % (a,n)
            subdf = df.query('alpha==%.2f and norm==%.1f' % 
                             (a,n))
            true_05 = subdf['truenpps_05'].mean()
            true_10 = subdf['truenpps_10'].mean()
            true_15 = subdf['truenpps_15'].mean()
            true_20 = subdf['truenpps_20'].mean()

            mean_05 = subdf['npps_05'].mean()/true_05
            lo_05 = subdf['npps_05'].quantile(.16)/true_05
            hi_05 = subdf['npps_05'].quantile(.84)/true_05
            poisson_05 = mean_05 / np.sqrt(subdf['n_05'].median())

            mean_10 = subdf['npps_10'].mean()/true_10
            lo_10 = subdf['npps_10'].quantile(.16)/true_10
            hi_10 = subdf['npps_10'].quantile(.84)/true_10
            poisson_10 = mean_10 / np.sqrt(subdf['n_10'].median())

            mean_15 = subdf['npps_15'].mean()/true_15
            lo_15 = subdf['npps_15'].quantile(.16)/true_15
            hi_15 = subdf['npps_15'].quantile(.84)/true_15
            poisson_15 = mean_15 / np.sqrt(subdf['n_15'].median())

            mean_20 = subdf['npps_20'].mean()/true_20
            lo_20 = subdf['npps_20'].quantile(.16)/true_20
            hi_20 = subdf['npps_20'].quantile(.84)/true_20
            poisson_20 = mean_20 / np.sqrt(subdf['n_20'].median())

            print 'r<0.5:',true_05,mean_05,lo_05,hi_05,poisson_05
            print 'r<1.0:',true_10,mean_10,lo_10,hi_10,poisson_10
            print 'r<1.5:',true_15,mean_15,lo_15,hi_15,poisson_15
            print 'r<2.0:',true_20,mean_20,lo_20,hi_20,poisson_20

            k+=1
            if plot:
                plt.subplot(len(alphas),len(norms),k)
                plt.errorbar([0.5,1,1.5,2],[mean_05,mean_10,mean_15,mean_20],
                             yerr=[[mean_05-lo_05,mean_10-lo_10,
                                    mean_15-lo_15,mean_20-lo_20],
                                   [hi_05-mean_05,hi_10-mean_10,
                                    hi_15-mean_15,hi_20-mean_20]],color='k',
                             ls='none',marker='o')
                plt.errorbar([0.5,1,1.5,2],[mean_05,mean_10,mean_15,mean_20],
                             yerr=[poisson_05,poisson_10,
                                   poisson_15,poisson_20],color='r',
                             ls='none')
                plt.axhline(1,color='k',ls=':')
                plt.annotate('a=%.1f\nn=%.1f' % (a,n),xy=(0.02,0.98),
                             xycoords='axes fraction',fontsize=10,va='top')
                plt.xlim((0,2.5))
                plt.ylim((0.5,1.5))
                plt.yticks([0.6,0.8,1.0,1.2,1.4])
                plt.xticks([0.5,1.0,1.5,2.0])

                ax = plt.gca()
                if k % len(norms) != 1:
                    plt.setp(ax.get_yticklabels(), visible=False)
                if k < len(norms)*(len(alphas)-1)+1:
                    plt.setp(ax.get_xticklabels(), visible=False)

                plt.draw()

def bias_test(N=1e4,npps=1.,p_tr=0.02,Nobs=3000,x1=1.,x2=2.,e1=0.,e2=1.,
              occ_quartiles=[1,1,1,1]):
    def e_fn(x):
        return e1 + (e2-e1)/(x2-x1) * (x-x1)

    eta_means = []
    f_ests = []

    for i in np.arange(N):
        n_true = int(Nobs*npps)
 
        n = rand.poisson(n_true*p_tr)
    
        ux = rand.random(size=n)

        occ_quartiles = np.array(occ_quartiles)
        occ_quartiles /= occ_quartiles.sum()

        dx = x2 - x1
        xq1 = x1 + dx/4
        xq2 = x1 + dx/2
        xq3 = x1 + 3*dx/4

        xs = x1 + ux*(x2-x1)
        es = e_fn(xs)
    
        uobs = rand.random(size=n)
        w_observed = np.where(uobs < es)
    
        f_est = (1/es[w_observed]).sum() / Nobs

        eta_means.append(es[w_observed].mean())
        f_ests.append(f_est)
        
    return np.array(f_ests),np.array(eta_means)
    
def bias_test_range(nfs=10):
    pass

class ZeroDetectionsError(Exception):
    pass
