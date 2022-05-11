"""
Created on Mon Feb  1 13:36:02 2016
@author: Skevja

This file contains definitions of the functions (not related to the model physics),
that are used later in the main code.
"""

import time
import os
import numpy as np
from collections import namedtuple
import scipy.signal as sgn
from astropy.table import Table 
from .constants import tp, tr
from . import localpath


def read_parameters(path_to_file):
    """
    Reads parameters from text file(s).
    
    :param path_to_file: Relative path to the parameter file(s). Main parameter file must have name 
        'parameters' and the second, optional parameter file, must be called 'sfrd_peaks_parameters'. 
    :type path_to_file: string
    
    :return: Names and values of parameters given in the parameter file(s).
    :rtype: namedtuple 
    """
    
    f = open(os.path.join('.',path_to_file),'r')
    paramd = {} 
    for line in f:
        try:
            # Read parameter name, commented lines are ignored.
            key, value = None, None
            key = line.split()[0] 
            if list(key)[0] == '#': pass
            else:
                # Read parameter value, decompose it into characters
                # (parameter values can be not only numbers, but also strings).
                value_str = line.split()[1] 
                value_str_char = list(value_str) 
                
                # Interpret input as a string, float or integer.
                if "'" in value_str_char:
                    value = str(value_str[1:-1])
                else:
                    if '.' in value_str_char:
                        value = float(value_str)
                    else:
                        value = int(value_str)    
                paramd[key]=value
        except:
            pass
    f.close
    
    if ('pkey' in paramd.keys()) and (paramd['pkey']==1) or (paramd['pkey']==2):
        message = ("File must have the following parameters organized as columns:\n"+
                  "sigmap[Msun/pc^2] : SF amplitude-related parameter (for R=Rsun)\n"+
                  "taup[Gyr]         : age of peak center\n"+
                  "dtaup[Gyr]        : dispersion in time\n"+
                  "Rp[kpc]           : peak center at R-axis\n"+
                  "dRp[kpc]          : dispersion in radius\n")                  
        if (paramd['pkey']==1):
            message += "sigp[km/s]      : W-velocity dispersion of the peak's populations\n"
        
        if os.path.isfile(os.path.join('.','sfrd_peaks_parameters')):
            sf_peaks = np.loadtxt(os.path.join('.','sfrd_peaks_parameters'))
            if len(sf_peaks)!=0:
                if len(sf_peaks[0])==6 or len(sf_peaks[0])==5:
                    paramd['sigmap'] = sf_peaks[:,0]
                    paramd['tpk'] = tp - sf_peaks[:,1]
                    paramd['dtp'] = sf_peaks[:,2]
                    paramd['Rp'] = sf_peaks[:,3]
                    paramd['dRp'] = sf_peaks[:,4]
                    if (paramd['pkey']==1):
                        paramd['sigp'] = sf_peaks[:,5]
                else:
                    print("\nSome of parameters are missing in 'sfrd_peaks_parameters'! " + message)
            else:
                print("\nFile 'sfrd_peaks_parameters' is empty, give the parameters! " + message)
        else: 
            print("\nFile 'sfrd_peaks_parameters' not found! Create it or copy to the directory. " + message)
        
    
    # All parameters are organized as a tuple and can be called as tuple_name.parameter_name
    # Non-local parameters
    pnames_nonlocal = ['Vsun','Rmin','Rmax','dR','Rd','Rt','Rg1','Rg2','Rg10','Rg20',
                       'fkey','Rf','Rdf','a_in','Mb','k_td2','k_dzeta','k_eta','Rp','dRp',
                       'k_FeHd0','k_FeHdp','k_rd','k_q']
                       
    pnames_amrd0 = ['FeHdp','rd','q','k_FeHd0','k_FeHdp','k_rd','k_q']
    pnames_amrd1 = ['rd1','rd2','Rbr1','Rbr2','Rbr3','k1_FeHdp','b1_FeHdp','k_alphaw','b_alphaw',
                    'k1_t01','b1_t01','k2_t01','b2_t01','k3_t01','b3_t01',
                    'k1_t02','b1_t02','k2_t02','b2_t02','k3_t02','b3_t02']
    
    if paramd['run_mode']==0:
        paramd_reduced = {}
        keys = list(paramd.keys())
        for i in range(len(keys)):
            if keys[i] not in pnames_nonlocal:
                paramd_reduced[keys[i]] = paramd[keys[i]]
        paramd = paramd_reduced
        
    pnames_imf = ['a0','a1','a2','a3','m0','m1','m2']
    if paramd['imfkey']==1:
        for i in range(len(pnames_imf)):
            del paramd[pnames_imf[i]]
    if paramd['fehkey']==0:
    	for i in range(len(pnames_amrd1)):    
            try:
                paramd[pnames_amrd1[i]]
            except: pass
    if paramd['fehkey']==1:
        try: 
            for i in range(len(pnames_amrd0)):
                del paramd[pnames_amrd0[i]]
        except: pass
    if paramd['fehkey']==2:
        try:
            paramd['FeHd0']
            for i in range(len(pnames_amrd0)):
                paramd[pnames_amrd0[i]]
        except: pass
        for i in range(len(pnames_amrd1)):
            try:
                del paramd[pnames_amrd1[i]]
            except: pass
    
    k,v = paramd.keys(), paramd.values()
    p = namedtuple('p',k) 
    p = p._make(v)   
    if paramd['pkey']==0:
        nparams = len(paramd.values())
    if paramd['pkey']==1:
        nparams = len(paramd.values()) + 6*len(paramd['sigmap']-1)
    if paramd['pkey']==2:
        nparams = len(paramd.values()) + 5*len(paramd['sigmap']-1)
    
    pnames_technical = ['run_mode','out_dir','out_mode','nprocess','pkey','imfkey','fkey','fehkey']
    count_technical = 0 
    keys = list(paramd.keys())
    for i in range(len(keys)):
        if keys[i] in pnames_technical:
            count_technical += 1
        
    print('\nParameter file(s) : ok.')
    print('Number of parameters = ',nparams, ', among them technical = ',count_technical)
    print('\n',p)    
        
    return p


def resave_parameters(path_to_parameterfile,path_to_parameterfile_copy,p):
    """
    Reads parameters from text file(s) and saves parameter file(s) 
    to the output folder with only those parameters which are needed for this run.
    
    :param path_to_file: Relative path to the parameter file(s).
    :type path_to_file: string 
    
    :return: None 
    """
    
    pnames_local = ['run_mode','out_dir','out_mode','nprocess','Rsun','zsun','Vsun','zmax','dz',
                    'sigmad','sigmat','sigmag1','sigmag2','sigmadh','sigmash',
                    'td1','td2','dzeta','eta','pkey','tt1','tt2','gamma','beta',
                    'FeHd0','FeHdp','rd','q','dFeHdt','n_FeHdt','FeHt0','FeHtp','rt','t0',
                    'FeHsh','dFeHsh','n_FeHsh','alpha','sige','sigt','sigdh','sigsh']                    
    if p.pkey==1:
        pnames_local.extend(['sigmap','taup','dtaup','sigp'])
    if p.pkey==2:
        pnames_local.extend(['sigmap','taup','dtaup'])
        
    # Main parameter file
    if p.pkey==0:
        f = open(os.path.join('.',path_to_parameterfile),'r')
        f_out = open(os.path.join('.',path_to_parameterfile_copy),'w')
    else:
        f = open(os.path.join('.',path_to_parameterfile[0]),'r')
        f_out = open(os.path.join('.',path_to_parameterfile_copy[0]),'w')

    for line in f:
        if line!="\n":
            # Read parameter name, commented lines are ignored.
            linechar = line.split()[0]
            if linechar[0]=='#':
                f_out.write(line)
            else:
                parameter = line.split('\t')[0]
                if p.run_mode==0:
                    if parameter in pnames_local:
                        f_out.write(line)
                else:
                    f_out.write(line)
        else: 
            f_out.write(line)     
    f.close
    f_out.close
    
    # SFRd-peaks parameter file
    if p.run_mode==0 and p.pkey!=0:
        f = open(os.path.join('.',path_to_parameterfile[1]),'r')
        f_out = open(os.path.join('.',path_to_parameterfile_copy[1]),'w')
        
        for line in f:
            if line!="\n":
                # Read parameter name, commented lines are ignored.
                linechar = line.split()[0]
                if linechar[0]=='#':
                    if linechar[:7]!='#sigmap':
                        f_out.write(line)
                    else:
                        parameters = line.split('\t')
                        parameter0 = list(parameters[0].split()[0])
                        parameters[0] = ''.join((parameter0[1:])) # removes '#' char
                        parameter1 = list(parameters[-1].split()[0])
                        parameters[-1] = ''.join((parameter1)) # removes '\n' char
                        ind_reduced = []
                        for i in range(len(parameters)):
                            if parameters[i] in pnames_local:
                                ind_reduced.append(i)   
                        ind_reduced = np.array(ind_reduced,dtype=np.int)
                        parameters = np.array(parameters)
                        f_out.write('#' + '\t'.join((parameters[ind_reduced])) + '\n')
                else:
                    parameters = line.split('\t')
                    parameters = np.array(parameters)
                    f_out.write('\t'.join((parameters[ind_reduced])))                  
            else: 
                f_out.write(line)     
                
        f.close
        f_out.close     
    


class LogFiles():
    """
    Class for text file manipulations. Includes methods for creating 
    text files and adding lines to them. Used for writing log files. 
    """

    def __init__(self,filename):
        """
        Creates an instance of the class with a given file name.
        
        :param filename: Name of the file. 
        :type filename: str 
        """
        self.filename = filename

    def write(self,text):
        """
        Creates a file, writes a line, closes the file.
        
        :param text: Text to be added to the file. 
        :type text: str 
        
        :return: None. 
        """
        f = open(self.filename,'w+')
        f.write(text)
        f.close()

    def append(self,text):
        """
        Opens the file, adds a line, closes the file.
        
        :param text: Text to be added to the file. 
        :type text: str 
        
        :return: None. 
        """
        f = open(self.filename,'a+')
        f.write(text)
        f.close()



class Timer():
    """
    Class for measuring time intervals. 
    """  
    
    def start(self):
        """
        Creates an instance and fixes current time.
        """
        return time.time()
        
    def stop(self,moment):
        """
        Returns time interval between the current and some previous 
        moment. Output in fractions of hours, minutes and seconds.
        
        :param moment: Time record (in the format of output of time.time()). 
        :type moment: float 
        
        :return: Time interval in hms. 
        :rtype: str 
        """
        
        s = round(time.time() - moment,2)
        h = int((s/3600)//1)
        mi = int(((s/3600-h)*60)//1)
        ss = round((((s/3600-h)*60)-mi)*60,3)
        return ''.join((str(h),'h ',str(mi),'m ',str(ss),'s'))
        
        
class ConvertAxes():
    """
    Interpolation methods to calculate values [new_axis1] for an 
    array [new_axis2], given [old_axis1] and [old_axis2]. 
    """  
         
    def interpolate(self,axis1_new,axis1_old,axis2_old):
        """
        Returns new axis2-values for a new set of axis1-values 
        given the old (axis1,axis2). Uses linear interpolation. 
        
        :param axis1_new: New values along axis1. 
        :type axis1_new: array-like
        :param axis1_old: Old values along axis1. 
        :type axis1_old: array-like
        :param axis2_old: Old values along axis2. 
        :type axis2_old: array-like
        
        :return: New values along axis2 corresponding to the new axis1-values set. 
        :rtype: 1d-array
        """ 
        
        pnum = 2            # Number of point used for interpolation 
        axis2_new = []      # Empty list for new axis2-values 
        
        # -----------------------------------------------------------------------------------------
        # We find two nearest values to a given new axis1-value in the old axis1-array.
        # Then we use these two neighbour points to perform linear interpolation. 
        # We manually set the slope to zero if points are too close.   
        # Then a new axis2-value is calculated. 
        # -----------------------------------------------------------------------------------------
        
        for i in range(len(axis1_new)):
            
            ind = np.argsort(abs(axis1_new[i]-axis1_old))[np.arange(pnum)]
            
            if abs(axis1_old[ind[0]]-axis1_old[ind[1]]) < 0.0000001: 
                k = 0 
            else:
                k = (axis2_old[ind[0]]-axis2_old[ind[1]])/(axis1_old[ind[0]]-axis1_old[ind[1]])
            b = axis2_old[ind[0]] - k*axis1_old[ind[0]]
            axis2_new.append(k*axis1_new[i]+b) 
            
        return np.array(axis2_new)
      
    
    def get_closest(self,axis1_new,axis1_old,axis2_old):
        """
        No interpolation is applied, the new axis1-values are 
        replaced by the closest values from the old axis1-array 
        and the corresponding old axis2-values are returned. This 
        method is good when the resolution of the old axis1-array 
        is much better than the resolution of the new axis1-array. 
        
        :param axis1_new: New values along axis1. 
        :type axis1_new: array-like
        :param axis1_old: Old values along axis1. 
        :type axis1_old: array-like
        :param axis2_old: Old values along axis2. 
        :type axis2_old: array-like
        
        :return: New values along axis2 corresponding to the new axis1-values set. 
        :rtype: 1d-array
        """  
        
        axis2_new = [] 
        for i in range(len(axis1_new)):
            ind = np.argsort(abs(axis1_new[i]-axis1_old))
            axis2_new.append(axis2_old[ind[0]]) 
            
        return np.array(axis2_new)
    
        
             
def rebin_histogram(bin_edges,x_centers,counts):
    """
    Uses existing histogram to create a new histogram for the 
    different set of x-bins, such that the overall counts are 
    conserved. 
    
    :param bin_edges: Edges of the new x-bins. 
    :type bin_edges: array-like
    :param x_centers: Centers of the old x-bins. 
    :type x_centers: array-like
    :param counts: Histogram counts corresponding to the old x-bin centers. 
    :type counts: array-like
    
    :return: New counts corresponding to the new x-bins (len(bin_edges)-1). 
    :rtype: 1d-array
    """  

    n = len(bin_edges) - 1                  # Number of new bins 
    bin_width = np.diff(bin_edges)[0]       # New bin width 
    dx = np.diff(x_centers)[0]              # Old bin width 
    hist = np.zeros((n)) 
    
    # ---------------------------------------------------------------------------------------------
    # Firstly, we select all old bins falling into a new bin, at least partially.
    # Then we calculate how far the outer boundary of the old bin lays within 
    # the range of the new bin, and assign weights to all selected old bins.
    # Case 1: Old bin not fully entered the new bin.
    # Case 2: Old bin partly left the new bin 
    # Case 3: For all old bins that are fully inside the new bin weights remain 1. 
    # ---------------------------------------------------------------------------------------------
    
    for i in range(n):
        
        ind = np.where((x_centers + dx/2 > bin_edges[i])&(x_centers - dx/2 <= bin_edges[i+1]))[0] 
        
        weights = np.linspace(1,1,len(ind))                                         # Case 3
        frac = x_centers[ind] + dx/2 - bin_edges[i] 
        weights[frac < dx] = frac[frac < dx]/dx                                     # Case 1
        weights[frac > bin_width] = (dx - frac[frac > bin_width] + bin_width)/dx    # Case 2
        
        hist[i] = np.sum(counts[ind]*weights) 

    return hist



def _transition_2curves_(epsilon,x_break,x,y):
    """
    This is a crazy routine of my own invention, that allows to 
    make a smooth `naturally looking` transition between two curves. 
    It could be a fragmet of cycloid or hyperbola, but according 
    to my tests, they don't always look good. This function is 
    applied only once in this package, in funcs.heffr and only if 
    the thin-disk flaring is swithed on. Then this routine makes 
    a smooth transition between the straight line and exponent. 
    If you know any better way to do this, you're welcome to 
    improve the code, but so far it is not really needed as this 
    function does the job. Just be careful if you decide to apply 
    this function for your own purposes, plot the outcome to check 
    what it does. 
    
    Parameters
    ----------
    epsilon : scalar
        Half-width of the window, where the smooth transition will 
        be constructed. Units of epsilon are the same as units of x. 
    x_break : scalar
        x-coordinate of a break point.
    x : array_like
        x-coordinates.
    y : array_like
        y-coordinates.

    Returns
    -------
    y_smooth : ndarray
        y-coordinates with a smooth transition in [x_break-epsilon, 
        x_break+epsilon].
    """
    
    if np.mean(np.diff(x)) > epsilon/2:
        dx = epsilon/10
        ind_br = np.where(abs((x-x_break))==np.amin(abs((x-x_break))))[0][0]
        
        x1 = np.arange(x[0],x_break,dx)
        x2 = np.arange(x_break,x[-1]+dx,dx)
        
        ax = ConvertAxes()
        y1 = ax.interpolate(x1,x[:ind_br],y[:ind_br])
        y2 = ax.interpolate(x2,x[ind_br:],y[ind_br:])
        
        X = np.concatenate((x1,x2),axis=0)
        Y = np.concatenate((y1,y2),axis=0)
    else:
        X = np.copy(x)
        Y = np.copy(y)
        
    x_smooth_start = x_break - epsilon
    x_smooth_stop = x_break + epsilon 
    
    if (x[0] > x_break - 3*epsilon):
        x_2en = x[0]
        print('Warning: (Rdf-Rmin)/epsilon < 3, smoothing algorithm may fail.\
              Decrease Rmin or epsilon for a given Rf.')
    else:
        x_2en = x_break - 3*epsilon
    
    if (x[-1] < x_break + 3*epsilon):
        x_2ep = x[-1] 
        print('Warning: (Rmax-Rdf)/epsilon < 3, smoothing algorithm may fail.\
              Increase Rmax or decrease epsilon for a given Rf.')
    else:
        x_2ep = x_break + 3*epsilon  
        
    ind_sm1 = np.where(abs((X-x_smooth_start))==np.amin(abs((X-x_smooth_start))))[0][0]
    ind_sm2 = np.where(abs((X-x_smooth_stop))==np.amin(abs((X-x_smooth_stop))))[0][0]

    ind_2en = np.where(abs((X-x_2en))==np.amin(abs((X-x_2en))))[0][0]
    ind_2ep = np.where(abs((X-x_2ep))==np.amin(abs((X-x_2ep))))[0][0]
    
    y_smooth_start = Y[ind_sm1]
    y_smooth_stop = Y[ind_sm2]
                  
    xx = np.concatenate((X[ind_2en:ind_sm1],X[ind_sm2:ind_2ep]),axis=-1)
    yy = np.concatenate((Y[ind_2en:ind_sm1],Y[ind_sm2:ind_2ep]),axis=-1)
 
    pp11 = np.poly1d(np.polyfit(xx,yy,2))
    pp12 = np.poly1d(np.polyfit(xx,yy,6))
    
    def pp1(xp):
        return np.mean([pp11(xp),pp12(xp)])
    
    smooth = [(x_smooth_stop-i)/2/epsilon*pp11(i) + (1 - (x_smooth_stop-i)/2/epsilon)*pp12(i)
              for i in X[ind_sm1:ind_sm2+1]]

    shift_low1 = smooth[0] - y_smooth_start
    shift_low2 = smooth[-1] - y_smooth_stop
    
    correction_low = np.zeros((ind_sm2-ind_sm1+1))
    correction_low[:int((ind_sm2-ind_sm1)/2)] =\
        [shift_low1*(i-X[int(ind_sm1+(ind_sm2-ind_sm1)/2)])/(x_smooth_start-\
        X[int(ind_sm1+(ind_sm2-ind_sm1)/2)]) for i in X[ind_sm1:int(ind_sm1+(ind_sm2-ind_sm1)/2)]]
    correction_low[int((ind_sm2-ind_sm1)/2):] =\
        [shift_low2*(i-X[int(ind_sm1+(ind_sm2-ind_sm1)/2)])/(x_smooth_stop-\
        X[int(ind_sm1+(ind_sm2-ind_sm1)/2)]) for i in X[int(ind_sm1+(ind_sm2-ind_sm1)/2):ind_sm2+1]]
   
    y_smooth = np.concatenate((Y[:ind_sm1],smooth-correction_low,Y[ind_sm2+1:]),axis=-1)
    if len(X)!=len(x):
        y_smooth = ax.interpolate(x,X,y_smooth)
                                       
    return y_smooth


def cumhist_smooth_savgol(y,m,n):
    """
    Smoothes a normalized cumulative distribution of some quantity 
    with the Savitzky-Golay filter. 
    
    :param y: y-coordinates of the normalized cumulative distribution.
    :type y: array-like
    :param m: Parameter window_length of scipy.signal.savgol_filter.
    :type m: int 
    :param n: Parameter polyorder of scipy.signal.savgol_filter.
    :type n: int 
    
    :return y_sm: Smoothed y.
    :rtype: 1d-array         
    """
    
    dn = 3*m
    y_long = np.concatenate((np.linspace(0,0,dn),y,np.linspace(1,1,dn)),axis=-1)
    y_sm = sgn.savgol_filter(y_long,m,n)
    
    y_sm = y_sm[dn:-dn]
    y_sm[y_sm<0] = 0 
    y_sm[y_sm>1] = 1
    
    return y_sm 


def _rotation_matrix_(axis, theta):
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def gauss_weights(x,mean,sigma):
    """
    Draws instances from the Gaussian PDF. 
    
    :param x: Set of points, where probability density (PD) has to be calculated.        
    :type x: array-like
    :param mean: Mean of the Gaussian distribution.
    :type mean: scalar 
    :param sigma: Standard deviation of the Gaussian distribution.
    :type sigma: scalar
        
    :return: PD values at x, normalized to unity.
    :rtype: array-like
    """  
    weights = [np.exp(-(i-mean)**2/2/sigma**2) for i in x]
    weights = weights/np.sum(weights)
    
    return weights


def reduce_table(tab,a):
    """
    Calculates the surface number density of the mono-age sub-populations, thus, 
    reduces the length of the 'stellar assemblies' table to the number of thin- or thick-disk, 
    or halo subpopulations. 
    
    :param tab: Isochrone columns.
    :type tab: dict 
    :param a: Collection of the fixed model parameters, useful quantities and arrays.
    :type a: namedtuple
    
    :return: Table with wo columns: 't' and 'N', Galactic time (Gyr) 
        and surface number densities (number/pc^2).
    :rtype: astropy Table 
    """
    
    timebins = np.arange(0,tp+tr,tr)
    time = np.subtract(tp,tab['age'])
    
    out = Table()
    out['t'], out['N'], out['Sigma'] = a.t, np.zeros((a.jd)), np.zeros((a.jd))
    
    c = 0 
    for i in range(a.jd):
        ind = np.where((time>=timebins[i]) & (time<timebins[i+1]))[0]
        c+=len(ind)
        if ind!=[]:
            out['N'][i] = np.sum(np.array(tab['N'][ind]))
            out['Sigma'][i] = np.sum(np.array(tab['N'][ind]*tab['Mf'][ind]))
        
    return out


def convolve2d_gauss(array,dxy_smooth,xy_range):
    """
    Smoothing a 2d-array with a Gaussian kernel.  

    :param array: Initial array.
    :type array: 2d-array.
    :param dxy_smooth: Dispersions in x and y, [dx,dy].
    :type dxy_smooth: list 
    :param xy_range: Min and max values along the array axis, 
        [[xmin,xmax],[ymin,ymax]].
    :type xy_range: list[list]
    
    :return: New smoothed array of the same shape.
    :rtype: 2d-array 
    """
    
    dx, dy = dxy_smooth
    [xmin,xmax],[ymin,ymax] = xy_range
    
    x_step = (xmax-xmin)/array.shape[0]
    y_step = (ymax-ymin)/array.shape[1]
    
    k_dx = int(round(dx/x_step,0))
    k_dy = int(round(dy/y_step,0))
    
    if k_dx%2==0:
        k_dx = k_dx - 1 
    if k_dy%2==0:
        k_dy = k_dy - 1 
    K = np.ones((k_dx,k_dy),np.float32)  
    K_norm = K/np.sum(K)
    array = sgn.convolve2d(array,K_norm,boundary='symm',mode='same')
    
    return array
    

def convolve1d_gauss(array,dx_smooth,x_range):
    """
    Smoothing a 1d-array with a Gaussian kernel.  

    :param array: Initial array.
    :type array: 1d-array.
    :param dx_smooth: Dispersion in x. 
    :type dx_smooth: scalar
    :param x_range:  Min and max values along the array axes, [xmin,xmax].
    :type x_range: list 
    
    :return: New smoothed array of the same shape.
    :rtype: 1d-array 
    """
    
    xmin, xmax = x_range
    x_step = (xmax-xmin)/len(array)
    
    k_dx = int(round(dx_smooth/x_step,0))
    
    if k_dx%2==0:
        k_dx = k_dx - 1 
    K = np.ones(k_dx,np.float32)  
    K_norm = K/np.sum(K)
    array = sgn.convolve(array,K_norm,mode='same')
    
    return array
    




