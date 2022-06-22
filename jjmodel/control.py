"""
Created on Thu Jan 28 18:58:04 2021

@author: skevja
"""

import sys
import numpy as np 
import inspect
from multiprocessing import cpu_count
from .constants import tp, tr
from .funcs import AVR, RadialDensity


def inpcheck_mode(n_options,mode,quantity,funcname,**kwargs):
    """
    Checks input of the functions defined below. 
    
    Parameters
    ----------
    n_options : int
        2 or 3, depending on what has to be analyzed: thin and 
        thick disk or thin, thick and thin+thick (total) disk.
    mode : str
        Specifies which component of the disk will be analyzed, 
        can be `d`, `t` or `dt` for the thin, thick or total disk, 
        respectively. If given wrong, falls back to `dt`. 
    quantity : str
        Name of quantity to be calculated.
    funcname : str
        Name of the function, for which the input is checked. 
    **kwargs : dict, optional keyword arguments
        merged : boolean
            If False, .

    Returns
    -------
    ln : TYPE
        DESCRIPTION.
    mode : TYPE
        DESCRIPTION.
    """
    
    ln = '$\mathrm{Thin \ disk}$'
    if mode!='d':
        if n_options==3:
            if mode=='t':
                ln = '$\mathrm{Thick \ disk}$'
            else:
                if mode!='dt':
                    mode='dt'
                    print(''.join(('\n',funcname,': Parameter `mode` can equal `d` (thin disk), ',
                                   '`t` (thick disk) or `dt` (total disk), check your input! ',
                                    'Falling back to the default and plotting total disk ',
                                    quantity,'.'))
                          )
                if 'merged' in kwargs and kwargs['merged']==False:
                    ln = '$\mathrm{Thin + thick \ disk}$'  
                else:
                    ln = '$\mathrm{Total \ disk}$'   
        else:
            if n_options!=2:
                print(''.join(('\n',inpcheck_mode.__name__,': Parameter `n_options` must be 2 ',
                      '(plots for thin disk or total disk) or 3 (thin, thick or total disk), ',
                      'check your intput!, Text on the plot may be incorrect.'))
                      )
            ln = '$\mathrm{Total \ disk}$'
            if mode!='dt':
                print(''.join(('\n',funcname,': Parameter `mode` can equal `d` (thin disk) ',
                               'or `dt` (total disk), check your input! Got ',str(mode),' instead. ',
                               'Falling back to the default and plotting total disk ',
                               quantity,'.'))
                     )
                mode='dt'
                if 'merged' in kwargs and kwargs['merged']==False:
                    ln = '$\mathrm{Thin + thick \ disk}$'  
    return (ln, mode)


def inpcheck_mode_comp(mode,mode_allowed,quantity,funcname,**kwargs):
    """
    
    Parameters
    ----------
    n_options : int
        3 or 4, depending on what has to be analyzed: thin and 
        thick disk and halo or thin, thick disk, halo and the 
        disk+halo (total) component.
    mode : str
        Specifies which component of the disk will be analyzed, 
        can be `d`, `t`, `sh` or `tot` for the thin, thick disk, 
        halo or total component, respectively. If given wrong, 
        falls back to `d`. 
    quantity : str
        Name of quantity to be calculated.
    funcname : str
        Name of the function, for which the input is checked. 
        
    Returns
    -------
    mode : TYPE
        DESCRIPTION.
    """
    
    this_function = inspect.stack()[0][3]
    mode_allowed_allowed = ['d','t','dt','sh','tot']
    mode_names={'d':'thin disk','t':'thick disk','sh':'stellar halo',
                'dt':'thin + thick disk','tot':'total'}
    mode_labels = {'d':'$\mathrm{Thin \ disk}$',
                   't':'$\mathrm{Thick \ disk}$',
                   'sh':'$\mathrm{Stellar \ halo}$',
                   'dt':['$\mathrm{Thin + thick \ disk}$','$\mathrm{Total \ disk}$'],
                   'tot':'$\mathrm{Total \ disk + halo}$'}
    test = True
    for i in range(len(mode_allowed)):
        if mode_allowed[i] not in mode_allowed_allowed:
            test=False
    if test==False:
        print("".join(("\n",this_function," in ",funcname, 
                       ": Parameter 'mode_allowed' can only include 'd'(thin disk), ",
                       "'t'(thick disk), 'sh'(stellar halo), 'dt'(thin+thick disk), ",
                       "or 'tot'(total), check your input! ",
                       "Falling back to the default mode_allowed=['d','t','dt','sh','tot'] for ",
                        quantity,'.')))    
    
    if mode not in mode_allowed:
        mode_allowed_str = mode_allowed[0]
        for i in range(len(mode_allowed)-1):
            mode_allowed_str += ', '+str(mode_allowed[i+1])
        print("".join(("\n",funcname,": Parameter 'mode_comp' must be one of [",
                       str(mode_allowed_str),"], but found '",mode,"' instead. ",
                       "Falling back to the default mode_comp='d' for the ",quantity,'.'))) 
        mode='d'
        
    ln = mode_labels[mode]
    if mode=='dt':
        if inpcheck_iskwargtype(kwargs,'merged',False,bool,this_function):
            ln = mode_labels[mode][0]
        else:
            ln = mode_labels[mode][1]
    
    return (ln, mode)


def inpcheck_radius(R,p,funcname):

    try:
        r_interval = [(i>=p.Rmin)&(i<=p.Rmax) for i in R]
        if (False in r_interval):
            print(''.join(('\n',funcname,': You use distances that are outside ',
                           'of the modelled interval [p.Rmin,p.Rmax], check your input! ',
                           'Too large or too short R are removed from the array.')))
            good = np.where(r_interval==True)[0]
            R = R[good]
    except:
        if R<p.Rmin or R>p.Rmax:
            print(''.join(('\n',funcname,': You use distance that is outside ',
                           'of the modelled interval [p.Rmin,p.Rmax], check your input! ',
                           'Falling back to the closest R available in the model.')))
            if R<p.Rmin:
                R = p.Rmin
            else:
                R = p.Rmax
    return float(R) 
            

def inpcheck_age(age,funcname):
    
    try:
        age_interval = [(i>=0)&(i<=tp) for i in age]
        if (False in age_interval):
            print(''.join(('\n',funcname,': You use ages that are outside ',
                           'of the modelled interval [0,tp], check your input! ',
                           'Too old or too young(negative?!) ages are removed from the array.')))
            good = np.where(np.array(age_interval)==True)[0]
            age = age[good]
    except:
        if age<0 or age>tp:
            print(''.join(('\n',funcname,': You use age that is outside ',
                           'of the modelled interval [0,tp], check your input! ',
                           'Falling back to the closest age available in the model.')))
            if age<0:
                age = 0
            else:
                age = tp
    return age 
    

def inpcheck_dz(dz,p,funcname):
    if dz < p.dz:
        print(''.join(('\n',funcname,': dz must be >= p.dz, see your parameter file. ',
                       'Falling back to the minial allowed dz=p.dz.')))
        dz = p.dz
    return dz


def inpcheck_height(z,p,funcname):
    
    if type(z[0])!=list:
        z_good = z
        z_interval = [(i>=0)&(i<=p.zmax) for i in z]
        if (False in z_interval):
            print(''.join(('\n',funcname,': You use heights that are outside ',
                           'of the modelled interval [0,p.zmax], check your input! ',
                           'Falling back to the closest z available in the model.')))
            if z_interval[0]==False and z_interval[1]==False:
                z_good = [0,p.zmax] 
            else:
                if z_interval[1]==False:
                    z_good[1] = p.zmax
                else:
                   z_good[0] = 0   
    else:
        z_good = [] 
        count = 0 
        for i in range(len(z)):
            z_interval = [(k>=0)&(k<=p.zmax) for k in z[i]]
            if (False in z_interval):
                count+=1
                if z_interval[0]==False and z_interval[1]==False:
                    pass
                else:
                    if z_interval[0]==False:
                        z_good.append([0,z[i][1]])
                    else:
                        z_good.append([z[i][0],p.zmax])
            else:
                z_good.append(z[i])
            
            if count!=0:   
                print(''.join(('\n',funcname,': You use heights that are outside ',
                               'of the modelled interval [0,p.zmax], check your input! ',
                               'Falling back to the closest z available in the model.')))
    return z_good


def inpcheck_mode_disk(mode,funcname):
    
    if mode!='face-on' and mode!='edge-on':
        print(''.join(('\n',funcname,': Parameter `mode` must be `face-on` or `edge-on`, ',
                       'check your input! Falling back to the default `face-on` option.')))
        mode = 'face-on'
        
    return mode
               

def inpcheck_kwargs(kwargs,valid_kwargs,funcname):
    keys_list = list(kwargs.keys())
    for i in range(len(keys_list)):
        if keys_list[i] not in valid_kwargs:
            print(''.join(("\n",funcname,": The key '",keys_list[i],
                           "' is not a valid keyword, check your input!")))
            raise KeyError('Wrong keyword name.')
        
        
def inpcheck_iskwargtype(kwargs,kwargname,kwargvalue,kwargtype,funcname):
    test = False
    if ((kwargname in kwargs) and (type(kwargs[kwargname])==kwargtype) 
        and (kwargs[kwargname]==kwargvalue)):
        test = True
    else: 
        if (kwargname in kwargs) and (type(kwargs[kwargname])!=kwargtype):
            print(''.join(('\n',funcname,': The key ',kwargname,' must be of type ',
                           str(kwargtype),', but given ',str(type(kwargs[kwargname])),'!')))
            raise TypeError('Wrong keyword type.')
    return test


def inpcheck_kwargs_compatibility(kwargs,funcname,**kwg):
    
    if inpcheck_iskwargtype(kwg,'plt_only',True,bool,funcname):
        if (('save_format' in kwargs) and ('save' not in kwargs or kwargs['save']==False)):
            print(funcname + ": Unnecessary input. Keyword 'save_format' works only when save=True.")
            
        if (inpcheck_iskwargtype(kwargs,'normalized',True,bool,funcname) and 
            inpcheck_iskwargtype(kwargs,'cumulative',True,bool,funcname)):
            print(funcname + ": Unnecessary input. When 'cumulative'=True, "+\
                          "the normalized cumulative mass profiles are plotted, "+\
                          "no need to specify normalized=True.")
    else:
        if ('tab' in kwargs) and ('mode_pop' in kwargs):
            print(funcname + ": Don't use keywords 'mode_pop' and 'tab' together.")
            raise KeyError("Invalid combination of parameters.")
        if ('ages' in kwargs) and ('mets' in kwargs):
            print(funcname + ": Don't use keywords 'ages' and 'mets' together.")
            raise KeyError("Invalid combination of parameters.")
        if (('mode_iso' in kwargs) and ('mode_pop' not in kwargs) and 
            (funcname!='hess_simple' and funcname!='pops_in_volume')):
            print(funcname + ": Unnecessary input. Keyword 'mode_iso' works only with 'mode_pop'.")
        if (('number' in kwargs) and kwargs['number']==True and 
            ('mode_pop' not in kwargs) and ('tab' not in kwargs)):
            print(funcname + ": Unnecessary input. Keyword 'number' works only with 'mode_pop' or 'tab'.")
        if (('tabname' in kwargs) and ('save' not in kwargs or kwargs['save']==False)):
            print(funcname + ": Unnecessary input. Keyword 'tabname' works only when save=True.")
        if (('save_format' in kwargs) and ('save' not in kwargs or kwargs['save']==False)):
            print(funcname + ": Unnecessary input. Keyword 'save_format' works only when save=True.")
        if (inpcheck_iskwargtype(kwargs,'normalized',True,bool,funcname) and 
            inpcheck_iskwargtype(kwargs,'cumulative',True,bool,funcname)):
            print(funcname + ": Unnecessary input. When 'cumulative'=True, "+\
                          "the normalized cumulative mass profiles are plotted, "+\
                          "no need to specify normalized=True.")
             

def inpcheck_sigwpeak(p,a):
    # Check the SF peaks' sigma-W parameters from file sfrd_peaks_parameters. 
    sigg1 = 3     
    tau0 = tp/((sigg1/p.sige)**(-1/p.alpha)-1)
    age_velocity = AVR()
    AVR0 = age_velocity.avr_jj10(a.t,tp,p.sige,tau0,p.alpha)
    npeak = len(p.sigmap)
    sigwpeak_excess = np.zeros((npeak))
    for i in range(npeak):
        indt = np.where(np.abs(a.t-p.tpk[i])==np.amin(np.abs(a.t-p.tpk[i])))[0][0]
        if p.sigp[i] < AVR0[indt]:
            message = "The value of parameter sigp = " + str(p.sigp[i]) + " km/s "+\
                      "from the file 'sfrd_peaks_parameters' is "+\
                      "smaller than AVR0(t), which is not allowed in this model. "+\
                      "Here t = tp - tauc, and tauc is a mean age corresponding to this SF peak "+\
                      "given in the same file. Increase the value of 'sigp'."                      
            print(message)
            raise ValueError("Invalid parameter value.")
        else:
            sigwpeak_excess[i] = np.sqrt(p.sigp[i]**2 - AVR0[indt]**2)
    return sigwpeak_excess



def reduce_kwargs(kwargs,allowed_kwargs):
    kwargs_calc = {}
    for key in kwargs:
        if key in allowed_kwargs:
            kwargs_calc[key] = kwargs[key]
    return kwargs_calc



def inpcheck_parameters(p):
    
    n_errors, n_warnings, n_notes = 0, 0, 0  
    errs, wrns, nts = '', '', ''
    
    # Solar parameters
    # -----------------------------------------------------------
    
    if p.Rsun <= 0:
        errs += "Parameter 'Rsun' must be positive.\n"
        n_errors += 1 
    
    if p.Rsun <= 8.117 or p.Rsun >= 8.239:
        wrns += "-> Got unexpected value for the Solar radius 'Rsun'. "+\
                "Check units (must be kpc). According to Gravity Collaboration et al. (2019), "+\
                "Rsun = 8.178 ± 0.013 (stat) ± 0.022 (sys) kpc.\n"
        n_warnings += 1 
    
    if p.zsun < 0 or p.zsun > 25:
        wrns += "-> Got unexpected value for the Solar height 'zsun'. "+\
                "Check units (must be pc). Expected positive value of ~0-25 pc.\n"
        n_warnings += 1 
        
    if p.run_mode!=0 and (p.Vsun < 0 or p.Vsun > 25): 
        wrns += "-> Got unexpected value for the Solar peculoar velocity 'Vsun'. "+\
                "Check units (must be km/s). Expected positive value of ~0-25 km/s "+\
                "(upper liimit is very approximate and probably very overestimated).\n"
        n_warnings += 1 
    
    # Thin-disk SFR
    # -----------------------------------------------------------
    
    if p.td1!=0:
        nts += "-> When parameter 'td1' is not equal to zero, "+\
               "start of the thin-disk formation is delayed by td1 Gyr.\n"
        n_notes += 1
        
    if p.td2 < 0:
        errs += "-> Got unexpected value for parameter 'td2', "+\
                "it must be positive. \n"
        n_errors += 1 
        
    if p.dzeta < -10 or p.dzeta > 10:
        wrns += "-> Got unexpected value for parameter 'dzeta'. "+\
                "This is SFR power index, do you really want to have sfr ~ t^(2*dzeta) with dzeta > 10 "+\
                "or dzeta < -10?. \n"
        n_warnings += 1 
        
    if p.eta < -10 or p.eta > 10:
        wrns += "-> Got unexpected value for parameter 'eta'. "+\
                "This is SFR power index, do you really want to have sfr ~ t^(-eta) with eta > 10 "+\
                "or eta < -10?. \n"
        n_warnings += 1 
                    
    if p.pkey!=0 and p.pkey!=1 and p.pkey!=2:
        errs += "-> Got unexpected value for parameter 'pkey'. "+\
                "Allowed options are 0 (no Gaussian peaks in SFR), "+\
                "1 (SFR with Gaussian peaks, and peaks' populations have special kinematics), "+\
                "2 (SFR with Gaussian peaks, and peaks' "+\
                "populations have thin-disk kinematics from AVR).\n"
        n_errors += 1 
        
    if p.run_mode!=0 and p.run_mode!=1:
        errs += "-> Got unexpected value for parameter 'run_mode'. "+\
                "Allowed options are 0 (local model) and "+\
                "1 (range of distances). \n"
        n_errors += 1 
    
    if p.fehkey!=0 and p.fehkey!=1 and p.fehkey!=2:
        errs += "-> Got unexpected value for parameter 'fehkey'. "+\
                "Allowed options are 0 (AMR based on Gaia locally, extension for other R arbitrary), "+\
                "1 (based on APOGEE at 4-14 kpc, parameters given manually), and "+\
                "2 (same as 1, but default parameters). \n"
        n_errors += 1 
        
    if p.fehkey==0:
        
        if p.FeHdp > 0.35 or p.FeHdp < 0:
            wrns += "-> Got unexpected value for the present-day thin-disk metallicity at Rsun 'FeHdp'. "+\
                    "Expected positive value of about ~0-0.35 dex "+\
                    "(depending on the AMR calibration, errors treatment...).\n"
            n_warnings += 1 
        
    if p.fehkey==1:
        
        if p.rd1 >= p.rd2:
            wrns += "-> AMR parameter 'rd1' must be less than 'rd2'. "+\
                    "Check your input.\n"
            n_warnings += 1 

        if p.Rbr1 >= p.Rbr2 or p.Rbr1 >= p.Rbr3 or p.Rbr3 <= p.Rbr2:
            errs += "->  ParametersThe following condition must be fulfilled:"+\
                    "'Rbr1' < 'Rbr2' < 'Rbr3'.\n"
            n_errors += 1 
            
    
    if p.pkey==1 or p.pkey==2:
        
        npeak = len(p.sigmap)
        if p.run_mode==1:
            R = np.arange(p.Rmin,p.Rmax+p.dR,p.dR)
            SR = RadialDensity(p.Rsun,p.zsun,R)
            SigmaR = SR.sigma_disk(p.sigmad,p.Rd)
                        
        for i in range(npeak):   
            
            if p.sigmap[i]==0:
                wrns += "-> Peak#"+str(i+1)+": Instead of setting parameter 'sigmap' to zero, "+\
                        "remove or comment the line in parameter file.\n"
                n_warnings += 1 
                    
            if p.sigmap[i]<0:
                errs += "->  Peak#"+str(i+1)+": SFR parameter 'sigmap' must be positive.\n"
                n_errors += 1 
            
            if (p.tpk[i] > tp + 3*abs(p.dtp[i])) or (p.tpk[i] < 0) or (p.tpk[i] > tp):
                errs += "->  Peak#"+str(i+1)+": Mean time (age) of a Gaussian peak at SFR is outside "+\
                        "of the modeled time range. Parameter 'taup' must be positive. \n"
                n_errors += 1 
            
            if p.dtp[i] <= 0:
                errs += "->  Peak#"+str(i+1)+": SFR parameter 'dtaup' must be positive.\n"
                n_errors += 1    
            
            if  p.sigmap[i] > p.sigmad:
                errs += "->  Peak#"+str(i+1)+": Value of SFR parameter 'sigmap' is too large, "+\
                        "you're trying to assign all mass to a Gaussian peak.\n"
                n_errors += 1    
                    
            if p.run_mode==1 and (p.Rmin < p.Rmax):
                
                indr = np.where(np.abs(p.Rp[i]-R)==np.amin(np.abs(p.Rp[i]-R)))[0][0]
                if  p.sigmap[i] > SigmaR[indr]:
                    errs += "->  Peak#"+str(i+1)+": Value of SFR parameter 'sigmap' is too large, "+\
                            "you're trying to assign all mass to a Gaussian peak.\n"
                    n_errors += 1    
                    
                if (p.Rp[i] < p.Rmin - 3*abs(p.dRp[i])) or (p.Rp[i] > p.Rmax  + 3*abs(p.dRp[i])):
                    wrns += "->  Peak#"+str(i+1)+": R-position of a Gaussian peak at SFR is outside of "+\
                            "the modeled distance range, such that it will not influence the model predictions.\n"
                    n_warnings += 1 
                    
                if p.dRp[i] < 0:
                    errs += "->  Peak#"+str(i+1)+": SFR parameter 'dRp' must be positive.\n"
                    n_errors += 1 
                            
        if p.pkey==1:
            sigg1 = 3 
            tau0 = tp/((sigg1/p.sige)**(-1/p.alpha)-1)
            t = np.arange(tr/2,tp+tr/2,tr)
            age_velocity = AVR()
            AVR0 = age_velocity.avr_jj10(t,tp,p.sige,tau0,p.alpha)     
            
            for i in range(npeak):
                if p.sigp[i] < 0:
                    errs += "->  Peak#"+str(i+1)+": Parameter 'sigp' must be positive.\n"
                    n_errors += 1 
                    
                indt = np.where(np.abs(t-p.tpk[i])==np.amin(np.abs(t-p.tpk[i])))[0][0]
                if p.sigp[i] < AVR0[indt]:
                    errs += "The value of parameter sigp = " + str(p.sigp[i]) + " km/s "+\
                              "from the file 'sfrd_peaks_parameters' is "+\
                              "smaller than AVR0(t), which is not allowed in this model. "+\
                              "Here t = tp - tauc, and tauc is a mean age corresponding to this SF peak "+\
                              "given in the same file. Increase the value of 'sigp'.\n"                      
                    n_errors += 1 
    
    # Thick-disk SFR
    # -----------------------------------------------------------
    if p.tt1 < 0:
        errs += "-> Thick-disk parameter 'tt1' must be positive.\n"
        n_errors += 1
    
    if p.tt2 > 5:
        wrns += "-> Your thick-disk formation duration 'tt2' is larger than 5 Gyr. "+\
                "Are you sure? Thick disk is a quickly formed MW component.\n"
        n_warnings += 1 
        
    if p.gamma < -10 or p.gamma > 10:
        errs += "-> Please, do not input crazy values for the thick-disk parameter 'gamma' :)\n"
        n_errors += 1
        
    if p.beta < 0 or p.beta > 10:
        errs += "-> Parameter 'beta' must be positive and not too large. If you want to supress "+\
                "thick-disk formation faster, decrease parameter 'tt2'. \n"
        n_errors += 1
    
    # IMF
    # -----------------------------------------------------------
    
    if p.imfkey!=0:
        nts += "-> You want to use a custom IMF. Don't forget to specify IMF parameters "+\
               "(can be added directly to the main parameter file).\n"
        n_notes += 1 
    
    if p.a0 < 0 or p.a0 > 5:
        errs += "-> IMF parameter 'a0' must be positive and not very large (upper limit is 5 here, "+\
                "which is also far too large). \n"
        n_errors += 1   
        
    if p.a1 < 0 or p.a1 > 5:
        errs += "-> IMF parameter 'a1' must be positive and not very large (upper limit is 5 here, "+\
                "which is also far too large). \n"
        n_errors += 1   
        
    if p.a2 < 0 or p.a2 > 5:
        errs += "-> IMF parameter 'a2' must be positive and not very large (upper limit is 5 here, "+\
                "which is also far too large). \n"
        n_errors += 1   
    
    if p.a3 < 0 or p.a3 > 5:
        errs += "-> IMF parameter 'a3' must be positive and not very large (upper limit is 5 here, "+\
                "which is also far too large). \n"
        n_errors += 1   
        
    if p.m0 < 0 or p.m0 > 100:
        errs += "-> Do not input negative or crazy values for the IMF parameter 'm0'.\n"
        n_errors += 1  
        
    if p.m1 < 0 or p.m1 > 100:
        errs += "-> Do not input negative or crazy values for the IMF parameter 'm1'.\n"
        n_errors += 1  
        
    if p.m2 < 0 or p.m2 > 100:
        errs += "-> Do not input negative or crazy values for the IMF parameter 'm2'.\n"
        n_errors += 1  
        
    if p.m0 >= p.m1 or p.m1 >= p.m2:
        errs += "-> IMF parameters 'm' must satisfy 'm0 < m1 < m2'.\n"
        n_errors += 1  
    
    # Metallicities
    # -----------------------------------------------------------
    
    if p.fehkey!=2 and (p.FeHd0 > -0.6 or p.FeHd0 < -0.9):
        wrns += "-> Got unexpected value for the initial thin-disk metallicity at Rsun 'FeHd0'. "+\
                "Expected value between -0.9 and -0.6 dex.\n"
        n_warnings += 1 
            
    if p.FeHt0 < -1.2 or p.FeHt0 > -0.7:
        wrns += "-> Got unexpected value for the initial thick-disk metallicity 'FeHt0'. "+\
                "Expected value between -1.2 and -0.7 dex.\n"
        n_warnings += 1 
    
    if p.FeHtp > 0.1 or p.FeHtp < -0.2:
        wrns += "-> Got unexpected value for the present-day thick-disk metallicity 'FeHtp'. "+\
                "Expected value between -0.2 and 0.1 dex "+\
                "(depending on the AMR calibration, errors treatment...).\n"
        n_warnings += 1 
        
    if p.fehkey==0 and p.FeHd0 > p.FeHdp: 
        errs += "-> Present-day thin-disk metallicity 'FeHdp' must be larger than "+\
                "the initial one 'FeHd0'.\n"
        n_errors += 1 
        
    if p.FeHt0 > p.FeHtp: 
        errs += "-> Present-day thick-disk metallicity 'FeHtp' must be larger than "+\
                "the initial one 'FeHt0'.\n"
        n_errors += 1 
    
    if p.t0 <= 0:
        errs += "-> Thick-disk AMR parameter 't0' must be positive.\n"
        n_errors += 1 
    
    if p.rt < 0:
        errs += "-> Thick-disk AMR parameter 'rt' must be positive.\n"
        n_errors += 1 
    
    if p.rt == 0:
        wrns += "-> When thick-disk AMR parameter 'rt' equals 0, thick disk has no chemical evolution (same metallicity for all times). "+\
                "Did you really mean that?\n"
        n_warnings += 1 
        
    if p.rt >= 5:
        wrns += "-> By setting large values for the thick-disk AMR parameter 'rt', you may produce unphysical models. "+\
                "Be careful, check the AMR plot.\n"
        n_warnings += 1 
    
    if p.fehkey==0 and p.q <= -1:
        errs += "-> Thin-disk AMR parameter 'q' must be > -1.\n"
        n_errors += 1 
    
    if p.fehkey==0 and p.rd < 0:
        errs += "-> Thin-disk AMR parameter 'rd' must be positive.\n"
        n_errors += 1 
        
    if p.fehkey==0 and p.rd >= 5:
        wrns += "-> By setting large values for the thin-disk AMR parameter 'rd', you may produce unphysical models. "+\
                "Be careful, check the AMR plot.\n"
        n_warnings += 1 
        
    if p.fehkey==0 and p.rd == 0:
        wrns += "-> When thin-disk AMR parameter 'rd' equals 0, thin disk has no chemical evolution (same metallicity for all times). "+\
                "Did you really mean that?\n"
        n_warnings += 1 
    
    if p.FeHsh > -1.1 or p.FeHsh < -1.9:
        wrns += "-> Got unexpected value for the mean halo metallicity 'FeHsh'. "+\
                "Expected value between -1.9 and -1.1 dex "+\
                "(assuming Gaussian distribution at -1.5 ± 0.4 dex).\n"
        n_warnings += 1 
            
    if p.dFeHsh < 0:
        errs += "-> Got unexpected value for the halo metallicity dispersion 'dFeHsh'. "+\
                "Must be positive.\n"
        n_errors += 1 
        
    if p.dFeHsh > 0.8:
        wrns += "-> Got unexpected value for the halo metallicity dispersion 'dFeHsh'. "+\
                "Expected value of ~0-0.8 dex.\n"
        n_warnings += 1 
            
    if p.n_FeHsh < 0:
        errs += "-> Got unexpected value for the number of mono-metallicity halo subpopulations 'n_FeHsh'. "+\
                "Must be positive uneven value.\n"        
        n_errors += 1 
        
    if p.n_FeHsh > 11:
        wrns += "-> Got unexpected value for the number of mono-metallicity halo subpopulations 'n_FeHsh'. "+\
                "Do you really need so many halo populations?\n"
        n_warnings += 1 
        
    if p.n_FeHsh%2==0:
        wrns += "-> Parameter 'n_FeHsh' should be uneven, better correct your input.\n "
        n_warnings += 1 
            
    if p.dFeHdt < 0:
        errs += "-> Got unexpected value for the dispersion of the disk "+\
                "mono-metallicity subpopulations 'dFeHdt'. Must be positive.\n"
        n_errors += 1 
        
    if p.dFeHdt > 0.2:
        wrns += "-> Got unexpected value for the dispersion of the disk "+\
                "mono-metallicity subpopulations 'dFeHdt'. "+\
                "Expected < ~0.2 dex. "+\
                "Think again whether you really want this large spread in AMR.\n"
        n_warnings += 1 
        
        if p.dFeHdt > 0.3:
            errs += "-> Do not input too large value for the dispersion of the disk "+\
                "mono-metallicity subpopulations 'dFeHdt'. "+\
                "Maximum allowed value is 0.3 dex.\n"
            n_errors += 1 
            
        
    if p.dFeHdt==0 and p.n_FeHdt>1:
        wrns += "-> You set the number of the disk mono-metallicity subpopulations 'n_FeHdt' "+\
                "larger than 1, but you don't allow any dispersion at fixed metallicity, "+\
                "dFeHdt=0. This is inconsistent, parameter n_FeHdt will not take effect. "+\
                "Increase your dFeHdt if you want to add some spread to AMR.\n"
        n_warnings += 1 
        
    if p.dFeHdt==0 and p.n_FeHdt<1:
        wrns += "-> Got unexpected value for parameter 'n_FeHdt', it must be equal to 1 when dFeHdt=0.\n"
        n_warnings += 1 
        
    if p.n_FeHdt < 1: 
        errs += "-> Got unexpected value for parameter 'n_FeHdt', it cannot be less than 1.\n"
        n_errors += 1 
            
    if p.dFeHdt!=0 and (p.n_FeHdt > 9 or p.n_FeHdt%2!=1 or p.n_FeHdt < 3):
        wrns += "-> Got unexpected value for the number of the disk mono-metallicity "+\
                "subpopulations 'n_FeHdt'. It should be uneven. "+\
                "Using nmetdt > 9 is not recommended as this will slow down the calculation a lot and "+\
                "produce very heavy stellar assembly tables. " +\
                "When dFeHdt > 0, n_FeHdt must be > 3 to sample the Gaussian PDF.\n"
        n_warnings += 1 
    
    # Radially extended model 
    # -----------------------------------------------------------
    
    if p.run_mode==1:                
        
        # Radial grid
    
        if p.Rmin < 4:
            errs += "-> Got unexpected value for parameter 'Rmin', "+\
                    "it must be larger or equal to 4 kpc, as the bulge zone is not included into "+\
                    "the current version of the JJ model.\n"
            n_errors += 1 
                
        if p.Rmax > 15:
            wrns += "-> Got unexpected value for parameter 'Rmax'. "+\
                    "Do you really want to model disk beyond ~15 kpc? At large distances "+\
                    "warp can strongly influence the disk vertical structure, and "+\
                    "warp is not included into the current version of the JJ model.\n"
            n_warnings += 1 
                
        if p.Rmin >= p.Rmax:
            errs += "-> Value of parameter 'Rmin' cannot be larger or equal 'Rmax'.\n"
            n_errors += 1 
            
        if p.dR <= 0 or p.dR > p.Rmax - p.Rmin:
            errs += "-> Parameter 'dR' must be positive and its maximum allowed value is (Rmax - Rmin) kpc.\n"                         
            n_errors += 1 
            
        if (p.Rmax - p.Rmin)/p.dR - int((p.Rmax - p.Rmin)/p.dR) != 0: 
            errs += "-> Parameter 'dR' is not consistent with the chosen range of Galactocentric"+\
                    "distances ['Rmin', 'Rmax']. An interval (Rmax - Rmin) must contain an int number of bins 'dR'.\n"                         
            n_errors += 1 
        
        # Scale lengths 
        
        if p.Rd < 1.8 or p.Rd > 3.2:
            errs += "-> Got unexpected value for the thin-disk scale length 'Rd'. "+\
                    "Expected value is ~2.5 ± 0.7 kpc.\n"
            n_errors += 1 
            
        if p.Rt < 1.5 or p.Rt > 2.5:
            errs += "-> Got unexpected value for the thick-disk scale length 'Rt'. "+\
                    "Expected value is ~2.0 ± 0.5 kpc.\n"
            n_errors += 1 
                
        if p.Rt > p.Rd:
            wrns += "-> You set thick-disk scale length 'Rt' larger than thin-disk scale length 'Rd', "+\
                    "did you mean it?\n"
            n_warnings += 1 
                
        if p.Rg1 < 2.0 or p.Rg1 > 4.0:
            wrns += "-> Got unexpected value for the molecular gas scale length 'Rg1'. "+\
                    "Expected value is ~3.0 ± 1 kpc.\n"
            n_warnings += 1 
                
        if p.Rg2 < 4.0:
            wrns += "-> Got unexpected value for the atomic gas scale length 'Rg2'. "+\
                    "Expected value larger than 4.0 kpc.\n"
            n_warnings += 1 
                
        if p.Rg1 > p.Rg2:
            errs += "-> You set the molecular gas scale length 'Rg1' larger than "+\
                    "the atomic gas scale length 'Rg2', which is inconsistent with observations.\n"                
            n_errors += 1 
            
        if p.Rg10 < 0: 
            errs += "-> Parameter 'Rg10', radius of the inner hole for molecular gas, "+\
                    "must be positive.\n"                
            n_errors += 1 
            
        if p.Rg10 > 4: 
            errs += "-> Parameter 'Rg10', radius of the inner hole for molecular gas, "+\
                    "is not allowed to be larger than 4 kpc (inconsistent with observations).\n"                
            n_errors += 1 
                        
        if p.Rg20 < 0: 
            errs += "-> Parameter 'Rg20', radius of the inner hole for atomic gas, "+\
                    "must be positive.\n"                
            n_errors += 1 
            
        if p.Rg20 > 4: 
            errs += "-> Parameter 'Rg20', radius of the inner hole for atomic gas, "+\
                    "is not allowed to be larger than 4 kpc (inconsistent with observations).\n"                
            n_errors += 1 
                
        if p.Rdf <= 0:
            errs += "-> Parameter 'Rdf' must be positive.\n"
            n_errors += 1 
            
        if p.Rf <= 6:
            errs += "-> Got unexpected value for the radius where thin-disk disk flaring starts 'Rf'. "+\
                    "Expected value larger than ~6.0 kpc.\n"
            n_errors += 1 
                
        if p.Rf <= p.Rmax:
            nts += "-> Value of parameter 'Rf' is smaller than 'Rmax', "+\
                   "i.e., you allow flaring of the thin disk.\n" 
            n_notes += 1 
        '''    
        if p.fkey==1:
            nts += "-> Value of parameter 'fkey' is 1, "+\
                   "i.e., you allow flaring of the thick disk.\n" 
            n_notes += 1 
            
        if p.fkey!=0 and p.fkey!=1:
            errs += "-> Got unexpected value for parameter 'fkey'. "+\
                    "Allowed options are 0 (no thick-disk flaring) and "+\
                    "1 (thick-disk flaring allowed).\n"
            n_errors += 1 
        '''        
        if p.a_in < -4 or p.a_in > -2:
            errs += "-> Got unexpected value for slope of the inner halo profile 'a_in'. "+\
                    "Expected value from -4 to -2.\n"
            n_errors += 1 
            
        if p.Mb < 0 or p.Mb > 5e11: 
            errs += "-> Got unexpected value for the bulge mass 'Mb'. "+\
                    "Cannot be negative or too large.\n"
            n_errors += 1 
        
        # Thin-disk SFR
        
        if (p.Rmin < p.Rmax):
            R = np.arange(p.Rmin,p.Rmax+p.dR,p.dR)
            td2_R = p.td2*(R/p.Rsun)**p.k_td2
            if True in (td2_R < 0):
                errs += "-> Got unexpected value for parameter 'td2' at some distance R, "+\
                        "'td2' must be positive, choose anouther value for 'k_td2'.\n"
                n_errors += 1 
            
            dzeta_R = p.dzeta*(R/p.Rsun)**p.k_dzeta
            if (True in (dzeta_R < -10)) or (True in (dzeta_R > 10)):
                wrns += "-> Got unexpected value for parameter 'dzeta' at some distance R, "+\
                        "'dzeta' should be not too small or too large (-10,10), "+\
                        "choose anouther value for 'k_dzeta'.\n"
                n_warnings += 1 
            
            eta_R = p.eta*(R/p.Rsun)**p.k_eta
            if (True in (eta_R < -10)) or (True in (eta_R > 10)):
                wrns += "-> Got unexpected value for parameter 'eta' at some distance R, "+\
                        "'eta' should be not too small or too large (-10,10), "+\
                        "choose anouther value for 'k_eta'.\n"
                n_warnings += 1 
            
    
    if p.zmax%p.dz != 0: 
        errs += "-> Parameter 'zmax' must contain an int number of vertical steps 'dz'.\n"
        n_errors += 1 
    
    if p.zmax < 1500:
        errs += "-> We do not recommend to use 'zmax' smaller than 1.5 kpc as otherwise vertical "+\
                "structure of the disk can be not properly reconstructed.\n"
        n_errors += 1 
            
    if p.zmax > 2000:
        errs += "-> The maximum height 'zmax' cannot be very large. It is not recommended to use the model "+\
                "at heights larger than 2000 pc.\n"
        n_errors += 1 
            
    if p.dz <=0 or p.dz > 20:
        errs += "-> Parameter 'dz' must be positive and not too large, please use something less than 20 pc.\n"                         
        n_errors += 1 
    
    # Vertical kinematics
    # -----------------------------------------------------------
    
    if p.alpha < 0.3 or p.alpha > 0.5:
        errs += "-> Got unexpected value for the AVR power index at Rsun 'alpha'. "+\
                "Expected value ~0.3-0.5.\n"
        n_errors += 1 
            
    if p.sige < 22 or p.sige > 31:
        errs += "-> Got unexpected value for the AVR parameter 'sige'. "+\
                "Expected value is 26 ± 5 km/s.\n"
        n_errors += 1 
            
    if p.sigt < 35 or p.sigt > 55:
        errs += "-> Got unexpected value for the thick-disk velocity dispersion 'sigt'. "+\
                "Expected value is in the range ~35-55 km/s.\n"
        n_errors += 1 
            
    if p.sigdh < 100 or p.sigdh > 200:
        errs += "-> Got unexpected value for the DM velocity dispersion 'sigdh'. "+\
                "Expected value is in the range ~100-200 km/s.\n"
        n_errors += 1 
            
    if p.sigsh < 70 or p.sigsh > 130:
        errs += "-> Got unexpected value for the halo velocity dispersion 'sigsh'. "+\
                "Expected value is in the range ~70-130 km/s.\n"
        n_errors += 1 
    
    # Local density normalizations 
    # -----------------------------------------------------------
    
    if p.sigmad < 15 or p.sigmad > 45:
        errs += "-> Got unexpected value for the thin-disk surface density at Rsun 'sigmad'. "+\
                "Expected value ~15-45 Msun/pc^2.\n"
        n_errors += 1 
        
    if p.sigmat < 1 or p.sigmat > 15:
        errs += "-> Got unexpected value for the thick-disk surface density at Rsun 'sigmat'. "+\
                "Expected value ~1-15 Msun/pc^2.\n"
        n_errors += 1 
    
    if p.sigmag1 < 1 or p.sigmag1 > 15:
        errs += "-> Got unexpected value for the molecular gas surface density at Rsun 'sigmag1'. "+\
                "Expected value ~1-15 Msun/pc^2.\n"
        n_errors += 1 
            
    if p.sigmag2 < 1 or p.sigmag2 > 30:
        errs += "-> Got unexpected value for the atomic gas surface density at Rsun 'sigmag2'. "+\
                "Expected value ~1-30 Msun/pc^2.\n"
        n_errors += 1 
            
    if p.sigmadh < 1 or p.sigmadh > 100:
        errs += "-> Got unexpected value for the DM surface density at Rsun 'sigmadh'. "+\
                "Expected value ~1-100 Msun/pc^2.\n"
        n_errors += 1 
            
    if p.sigmash < 0.001 or p.sigmash > 5:
        errs += "-> Got unexpected value for the halo surface density at Rsun 'sigmash'. "+\
                "Expected value ~0.001-5 Msun/pc^2.\n"
        n_errors += 1 
    
    # Technical staff
    # -----------------------------------------------------------
    
    if p.nprocess < 1 or p.nprocess > cpu_count():
        errs += "-> Got unexpected value for the number of CPU cores 'nprocess'. "+\
                "Minimal value is 1, maximal value is the number of CPU cores of your computer "+\
                "(you have " + str(cpu_count()) + "). "+\
                "When nprocess is larger than the number of CPU cores, calculation is slower.\n"                         
        n_errors += 1 
            
    if type(p.out_dir)!=str:
        errs += "-> Parameter 'out_dir' must be string.\n"
        n_errors += 1 
        
    if p.out_mode!=0 and p.out_mode!=1:
        errs += "-> Got unexpected value for parameter 'out_mode'. "+\
                "Allowed options are 0 (clean existing directory before saving the output) and "+\
                "1 (overwrite files in the existing directory).\n"
        n_errors += 1 
        
    # Output
    # -----------------------------------------------------------
        
    print("\nParameters checked: "+str(n_notes)+" Reminders, "+str(n_warnings)+\
          " Warnings, "+str(n_errors)+" Errors.\n")
    
    if n_notes!=0:
        print("\nKeep in mind.\n"+nts)
    if n_warnings!=0:
        print("\nWarnings.\n"+wrns)
    if n_errors!=0:
        raise ValueError('\n'+errs+'Correct your input.')



class CheckIsoInput():
    """
    Collection of functions that catch (some of) the user's 
    mistakes in the input to the isochrone-related functions, and 
    give helpful suggestions.
    """
    
    def check_mode_isochrone(self,mode,funcname):
        """
        Checks whether parameter `mode` is correct, when type of 
        isochrone set is chosen.
        
        Parameters
        ----------
        mode : str
            Defines which set of isochrones is used, can be 'Padova', 
            'MIST', or 'BaSTI'. 
        funcname : str
            Name of the checked function. 
        """  
        
        if mode!='Padova' and mode!='MIST' and mode!='BaSTI':
            print(''.join(('\n',funcname,": Parameter 'mode' ",
                  "can be 'Padova', 'MIST' or 'BaSTI', check your input!")))
            
                    
    def check_mode_component(self,mode,funcname):
        """
        Checks whether parameter `mode` is correct, when type of 
        the Galactic component is chosen.
        
        Parameters
        ----------
        mode : str
            Defines which Galactic component is simulated, can be 
            `d`, `t` or `dt` (thin, thick, and total disk = thin + 
            thick, respectively). 
        funcname : str
            Name of the checked function.
        """  
        
        if mode!='t' and mode!='d' and mode!='sh':
            print(''.join(('\n',funcname,': Parameter `mode`',
            ' must be `d`(thin disk) or `t`(thick disk), or `sh`(stellar halo), check your input!')))
                    
                
    def check_photometric_system(self,photometric_system,funcname,**kwargs):
        """
        Checks whether parameter `photometric_system` is correct.
        
        Parameters
        ----------
        photometric_system : str
            Name of the photometric system to use, can be `UBVRIplus`, 
            `GaiaDR2_MAW`, `GaiaEDR3`, `UBVRIplus+GaiaDR2_MAW`,
            `UBVRIplus+GaiaEDR3`, `GaiaDR2_MAW+GaiaEDR3`. 
            For MIST isochrones UBVRIplus = UBV(RI)c + 2MASS, for 
            Padova UBVRIplus = UBVRIJHK. 
        funcname : str
            Name of the checked function. 
        **kwargs : dict, optional keyword arguments 
            print : boolean
                If True, prints the list of photometric systems. 
        """  
        
        photo_dict = {1:'UBVRIplus',
                      2:'GaiaDR2_MAW',
                      3:'GaiaEDR3',
                      4:'UBVRIplus+GaiaDR2_MAW',
                      5:'UBVRIplus+GaiaEDR3',
                      6:'GaiaDR2_MAW+GaiaEDR3',
                      7:'UBVRIplus+GaiaDR2_MAW+GaiaEDR3'}
        
        if 'print' in kwargs and kwargs['print']==True:
            print(photo_dict)
            
        if type(photometric_system)!=int or (photometric_system<1 or photometric_system>7):
            message = funcname + ": Parameter 'photometric_system' " +\
                      "must be integer in the range of [1,7], check your input!\n" + str(photo_dict)
            print(message)
            raise ValueError("Invalid parameter value.")
        else: 
            bands = photo_dict[photometric_system]
        return bands
                                
                    
    def check_metallicity(self,met,metmin,metmax,funcname,**kwargs):
        """
        Checks whether the given metallicity value lays within the 
        available interval.
        
        Parameters
        ----------
        met : [Fe/H] or Z
            Metallicity. 
        metmin, metmax : [Fe/H] or Z 
            Minimum and maximum metallicities available. 
        """  
        test = True
        if (met > metmax) or (met < metmin):  
            test = False
            if inpcheck_iskwargtype(kwargs,'print_warning',True,bool,funcname):
                print(''.join(('\n',funcname,': Given metallicity [Fe/H] =',
                               str(round(met,2)),' is',' outside the available metallicity interval [',
                               str(round(metmin,2)),',',str(round(metmax,2)),
                               '], selected best isochrone may be not representative.')))
        return test
            



            
            
