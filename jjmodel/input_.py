"""
Created on Wed Mar  2 11:10:52 2016
@author: Skevja

Script to read model parameters from parameterfile 
and set the input model functions.  
"""

import numpy as np
from collections import namedtuple
from .tools import read_parameters,Timer
from .funcs import SFR, AMR, hgr, RadialDensity, RotCurve
from .constants import tp, tr, ZN, PC, YR, HR, KM, pm_SgrA
from .iof import dir_tree
from .control import inpcheck_sigwpeak, inpcheck_parameters


timer = Timer()
t_start = timer.start()

# ==================================================================================================
# Input parameters, useful indices and arrays
# ==================================================================================================

p = read_parameters('./parameters')
inpcheck_parameters(p)
# -------------------------------------------------------------------------------------------------
# jd and jt are number of the thin- and thick-disk subpopulations. 
# We also introduce the time- and age-arrays. Analogically, Rbins is the number of radial bins.
# dzmax corresponds to the maximum value of the normalized height z (depends on zmax)
# ddz is the normalized z-step, n is a number of z-bins. 
# dzeq is a normalized z-grid with equal step in linear space. 
# -------------------------------------------------------------------------------------------------

jd, jt = int((tp-p.td1)/tr), int(p.tt2/tr) 
jd_array, jt_array = np.arange(jd), np.arange(jt) 
t = np.arange(tr/2,tp+tr/2,tr)
tau = np.subtract(tp,t)

if p.run_mode!=0:
    Rbins = int((p.Rmax-p.Rmin)/p.dR+1) 
    R_array = np.arange(Rbins)
    R = np.arange(p.Rmin,p.Rmax+p.dR,p.dR)
    R_edges = np.concatenate((R-p.dR/2,[R[-1]+p.dR/2]))

dzmax = p.zmax/ZN 
ddz = p.dz/ZN 
n = int(p.zmax/p.dz) 
dzeq = np.arange(ddz/2,dzmax+ddz/2,ddz)
z = np.arange(p.dz/2,p.zmax+p.dz/2,p.dz)
n_array = np.arange(n) 

T = dir_tree(p)     # Directory tree where all output will be stored. 

# For convenience, we group these quantities into a touple.
q_names = ['t','tau','jd','jt','n','jd_array','jt_array','n_array','dzmax','ddz','dzeq','z','T']
q_arrays = [t,tau,jd,jt,n,jd_array,jt_array,n_array,dzmax,ddz,dzeq,z,T]
if p.run_mode!=0:
    qr_names = ['R','Rbins','R_array','R_edges']
    qr_arrays = [R,Rbins,R_array,R_edges]
    q_names = q_names + qr_names
    q_arrays = q_arrays + qr_arrays
                    
a = namedtuple('a',q_names)            
a = a._make(q_arrays)

# ==================================================================================================
# Input functions
# ==================================================================================================

# Gas scale heights from Nakanishi and Sofue (2016)
# Indices 1 and 2 correspond to the molecular and atomic gas, respectively. 
hg10, hg20 = hgr(p,a)
    
# Age-metallicity relation of the thin and thick disk:
amr = AMR()
if p.fehkey==0:
    amrt = amr.amrt_sj21(a.t,p.t0,p.rt,p.FeHt0,p.FeHtp)
else:
    amrt = amr.amrt_sj22(a.t,p.t0,p.rt,p.FeHt0,p.FeHtp)
if p.fehkey==0:
    amrd0 = amr.amrd_sj21(a.t,tp,p.q,p.rd,p.FeHd0,p.FeHdp)
if p.fehkey==1:
    amrd0 = amr.amrd_global_sj22_custom(a.t,p.Rsun,p)
if p.fehkey==2:
    amrd0 = amr.amrd_global_sj22_default(a.t,p.Rsun,p)
    

# Mass-loss function based on the adopted AMR. 
gd0 = amr.mass_loss(a.t,amrd0)
gt = amr.mass_loss(a.t,amrt)[:a.jt]


# Star formation rate function based on the calculated mass loss function.
sfr = SFR()
SFRt0, NSFRt0 = sfr.sfrt_sj21(a.t[:a.jt],p.gamma,p.beta,p.tt1,p.tt2,p.sigmat,g=gt)
if p.pkey==0:
    SFRd0, NSFRd0 = sfr.sfrd_sj21(a.t,p.dzeta,p.eta,p.td1,p.td2,p.sigmad,g=gd0) 
else:
    SFRd0, NSFRd0, Fp0 = sfr.sfrd_sj21_multipeak(tp,tr,a.t,p.dzeta,p.eta,p.td1,p.td2,p.sigmad,
                                                 p.sigmap,p.tpk,p.dtp,g=gd0)                                                     
SFRtot0 = np.concatenate((np.add(SFRd0[:a.jt],SFRt0),SFRd0[a.jt:]),axis=None)
NSFRtot0 = SFRtot0/np.mean(SFRtot0)
if p.pkey==1:
    sigwpeak_excess = inpcheck_sigwpeak(p,a)

if p.run_mode!=0:
    
    # Derivation of DM scaling parameter 
    # (such that predicted Vc at Rsun is consistent with assumed Rsun, Vsun and pm_SgrA)
    Vphi_sun = p.Rsun*PC*np.tan(np.deg2rad(pm_SgrA/KM/YR/HR))
    Vc_sun = Vphi_sun - p.Vsun
    rhodh0 = 0.014          # Msun/pc^3, expected DM density at Rsun
    rhosh0 = 0.00014        # Msun/pc^3, expected halo density at Rsun
    ah = 5                  # kpc, Initial value of DM scaling parameter
    vc0_diff = 1            # km/s, Initial value of discrepancy 
    vc0_eps = 0.1           # km/s, Precision of Vc
    
    RC = RotCurve(p.Rsun,np.array([p.Rsun]))
    vcd = RC.vc_disk(p.sigmad,p.Rd,0)
    vct = RC.vc_disk(p.sigmat,p.Rt,0)
    vcg1 = RC.vc_disk(p.sigmag1,p.Rg1,p.Rg10)
    vcg2 = RC.vc_disk(p.sigmag2,p.Rg2,p.Rg20)
    vcb = RC.vc_bulge(p.Mb)
    #vcsh = RC.vc_halo_power_law(rhosh0,-p.a_in)    
    
    while vc0_diff > vc0_eps:
        vcdh = RC.vc_halo_cored_iso_sphere(rhodh0,ah)
        vc = RC.vc_tot([vcb,vcd,vct,vcg1,vcg2,vcdh])
        Vc0 = RC.vc0(vc)    
        vc0_diff = np.abs(Vc_sun - Vc0)
        ah = ah*Vc0/Vc_sun
        #print(vc0_diff,ah,Vc0)

    (hg1,hg10), (hg2,hg20) = hgr(p,a)
    
    # Radial surface density profiles of the Galactic components. 
    SR = RadialDensity(p.Rsun,p.zsun,R)
    SigmaR = np.array([SR.sigma_disk(p.sigmad,p.Rd),                             # thin d.
                       SR.sigma_disk(p.sigmag1,p.Rg1),                           # mol. gas 
                       SR.sigma_disk(p.sigmag2,p.Rg2),                           # at. gas
                       SR.sigma_disk(p.sigmat,p.Rt),                             # thick d.
                       SR.sigma_dm_halo(p.zmax*1e-3,p.sigmadh,ah),               # DM
                       SR.sigma_stellar_halo(p.zmax*1e-3,p.sigmash,p.a_in)])     # Halo

    # Age-metallicity relation of the thin and thick disk. 
    amr = AMR()
    amrd = amr.amrr(p,a)

    # Mass-loss function based on the adopted AMR. 
    gd = np.zeros((a.Rbins,a.jd))
    for i in range(a.Rbins):
        gd[i] = amr.mass_loss(a.t,amrd[i])

    # Star formation rate function based on the calculated mass loss function.
    sfr = SFR()
    if p.pkey==0:
        (SFRd,NSFRd,SFRd0,NSFRd0),(SFRt,NSFRt,SFRt0,NSFRt0),(SFRtot,NSFRtot,SFRtot0,NSFRtot0) =\
            sfr.sfrr(p,a,gd,gd0,gt)
    else:
        (SFRd,NSFRd,Fp,SFRd0,NSFRd0,Fp0),(SFRt,NSFRt,SFRt0,NSFRt0),(SFRtot,NSFRtot,SFRtot0,NSFRtot0) =\
            sfr.sfrr(p,a,gd,gd0,gt)
        if p.pkey==1:
            sigwpeak_excess = inpcheck_sigwpeak(p,a)
        
    
# Again, for the future use, we create a single object (dict) containing defined functions. 
inp_names = ['SFRd0','NSFRd0','SFRt0','NSFRt0','SFRtot0','NSFRtot0','gd0','gt','AMRd0','AMRt','hg10','hg20']
inp_arrays = [SFRd0,NSFRd0,SFRt0,NSFRt0,SFRtot0,NSFRtot0,gd0,gt,amrd0,amrt,hg10,hg20]
if p.pkey!=0:
    inp_names.extend(['Fp0'])
    inp_arrays.extend([Fp0])
    if p.run_mode!=0:
        inp_names.extend(['Fp'])
        inp_arrays.extend([Fp])
    if p.pkey==1:
        inp_names.extend(['sigwpeak_excess'])
        inp_arrays.extend([sigwpeak_excess])

if p.run_mode==1:
    inpr_names = ['SFRd','NSFRd','SFRt','NSFRt','SFRtot','NSFRtot','gd','AMRd',
                  'SigmaR','hg1','hg2','ah']
    inpr_arrays = [SFRd,NSFRd,SFRt,NSFRt,SFRtot,NSFRtot,gd,amrd,SigmaR,hg1,hg2,ah]
    if p.pkey!=0:
        inpr_names.extend('Fp')
        inpr_arrays.extend(Fp)
    inp_names = inp_names + inpr_names
    inp_arrays = inp_arrays + inpr_arrays

inp = {}
for i in range(len(inp_names)):
    inp[inp_names[i]] = inp_arrays[i]
    
                            
print('Configuration finished: ',timer.stop(t_start))




