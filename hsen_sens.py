#Kyle Gourlie
#7/23/2024
#library imports
import os, h5py, pickle, glob, json, gc
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib as mpl
import healpy as hp
import astropy.units as u
import astropy.constants as c
from enterprise.pulsar import Pulsar as ePulsar

import sensitivity as hsen
import sim as hsim
import skymap as hsky

 
class PseudoPulsar:
    """Quick class to store data from HDF5 file in prep for hasasia pulsar creation"""
    def __init__(self, toas, toaerrs, phi, theta, pdist, N, Mmat=None):
        self.N = N
        self.Mmat = Mmat
        self.phi = phi
        self.theta = theta
        self.toas = toas
        self.toaerrs = toaerrs
        self.pdist = pdist

class PseudoSpectraPulsar:
    """Quick class to store data from HDF5 in prep for hasasia spectrum pulsar creation"""
    def __init__(self, toas, toaerrs, phi, theta, pdist, K_inv, G, designmatrix):
        self.K_inv = K_inv
        self.G = G
        self.phi = phi
        self.theta = theta
        self.toas = toas
        self.toaerrs = toaerrs
        self.pdist = pdist
        self.designmatrix = designmatrix

def get_psrname(file,name_sep='_'):
    """Function that grabs names of pulsars from parameter files
    
    Returns:
        Pulsar name
    """
    return file.split('/')[-1].split(name_sep)[0]

def make_corr(psr: ePulsar, noise:dict, yr:float)->np.array:
    """_summary_: Computes white noise correlation matrix for a given enterprise.pulsar object

    Args:
        - psr (ePulsar): enterprise.pulsar object
        - noise (dict): white noise parameters with front and backends
        - yr (float): if yr=15, then change key_eq and sigma_sqt

    Returns:
        np.array: white noise correlation matrix
    """
    N = psr.toaerrs.size
    corr = np.zeros((N,N))
    _, _, fl, _, bi = hsen.quantize_fast(psr.toas,psr.toaerrs,
                                         flags=psr.flags['f'],dt=1)
    keys = [ky for ky in noise.keys() if psr.name in ky]
    backends = np.unique(psr.flags['f'])
    sigma_sqr = np.zeros(N)
    ecorrs = np.zeros_like(fl,dtype=float)
    for be in backends:
        mask = np.where(psr.flags['f']==be)
        key_ef = '{0}_{1}_{2}'.format(psr.name,be,'efac')
        if yr == 15:
            key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,'t2equad')
            sigma_sqr[mask] = (noise[key_ef]**2 * ((psr.toaerrs[mask]**2) ## t2equad -- new/correct for 15yr
                           + (10**noise[key_eq])**2))
        else:
            key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,'equad')
            sigma_sqr[mask] = (noise[key_ef]**2 * (psr.toaerrs[mask]**2) ## tnequad -- old/wrong but used in 15yr
                            + (10**noise[key_eq])**2)
        mask_ec = np.where(fl==be)
        key_ec = '{0}_{1}_log10_{2}'.format(psr.name,be,'ecorr')
        ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise[key_ec])
    j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
         for ii, bucket in enumerate(bi)]

    J = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J
    return corr

def enterprise_creation(pars:list, tims:list, ephem:str)->list:
    """_summary_: Generate list of enterprise.pulsars objects

    Args:
        pars (list): list of parameter files 
        tims (list): list of timing files
        ephem (str): ephemeris

    Returns:
        list: list of enterprise.pulsar objects
    """
    enterprise_Psrs = []
    count = 1
    for par,tim in zip(pars,tims):
        if count <= kill_count:
            if ephem=='DE440':
                ePsr = ePulsar(par, tim,  ephem=ephem, timing_package='pint')
            else:
                ePsr = ePulsar(par, tim,  ephem=ephem)
            enterprise_Psrs.append(ePsr)
            print('\rPSR {0} complete'.format(ePsr.name),end='',flush=True)
            print(f'\n{count} pulsars created')
            count +=1
        else:
            break
    return enterprise_Psrs

def enterprise_pickle(ePsrs:list, pickle_dir):
    with open(pickle_dir, 'wb') as f:
        pickle.dump(ePsrs, f)

def pickle_enterprise(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        ePsrs = pickle.load(f)
        return ePsrs

def enterprise_hdf5(ePsrs:list, noise:dict, yr:float, edir:str):
    """Writes enterprise.pulsar objects onto HDF5 file with WN Covariance matrix attributes.

    - ePsrs (list): List of enterprise.pulsar objects
    - edir: Directory in which to store HDF5 file under
    """
    Tspan = hsen.get_Tspan(ePsrs)
    with h5py.File(edir, 'w') as f:
        Tspan_h5 = f.create_dataset('Tspan', (1,), float)
        Tspan_h5[:] = Tspan
        #numpy array stored with placeholders so names can be indexed into it later, also storing strings as bytes
        name_list = np.array(['X' for _ in range(len(ePsrs))], dtype=h5py.string_dtype(encoding='utf-8'))
        #pseudo while/for-loop designed to delete first entry
        i = 0
        while True:
            print(f'{ePsrs[0].name}\t{ePsrs[0].toas.size}\n')
            if thin_val == None:
                if ePsrs[0].toas.size >= 20_000:
                    ePsrs[0].thin = 5
                else:
                    ePsrs[0].thin = 1
            else:
                ePsrs[0].thin = thin_val

            ePsrs[0].N = make_corr(ePsrs[0], noise, yr)[::ePsrs[0].thin, ::ePsrs[0].thin]
            hdf5_psr = f.create_group(ePsrs[0].name)
            hdf5_psr.create_dataset('toas', ePsrs[0].toas[::ePsrs[0].thin].shape, ePsrs[0].toas[::ePsrs[0].thin].dtype, data=ePsrs[0].toas[::ePsrs[0].thin])
            hdf5_psr.create_dataset('toaerrs', ePsrs[0].toaerrs[::ePsrs[0].thin].shape,ePsrs[0].toaerrs[::ePsrs[0].thin].dtype, data=ePsrs[0].toaerrs[::ePsrs[0].thin])
            hdf5_psr.create_dataset('phi', (1,), float, data=ePsrs[0].phi)
            hdf5_psr.create_dataset('theta', (1,), float, data=ePsrs[0].theta)
            hdf5_psr.create_dataset('designmatrix', ePsrs[0].Mmat[::ePsrs[0].thin,:].shape, ePsrs[0].Mmat[::ePsrs[0].thin,:].dtype, data=ePsrs[0].Mmat[::ePsrs[0].thin,:], compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('N', ePsrs[0].N.shape, ePsrs[0].N.dtype, data=ePsrs[0].N, compression="gzip", compression_opts=compress_val)
            hdf5_psr.create_dataset('pdist', (2,), float, data=ePsrs[0].pdist)
            name_list[i] = ePsrs[0].name
            f.flush()
            del ePsrs[0]
            i+=1
            #once all the pulsars are deleted, the length of the list is zero
            if len(ePsrs) == 0:
                break
            
        f.create_dataset('names',data = name_list)
        f.flush()
        print('enterprise.pulsars successfully saved to HDF5\n')



def hsen_pulsar_entry(psr:hsen.Pulsar, dir:str):
    """Writes hasasia pulsar object to hdf5 file"""   
    with h5py.File(dir, 'a') as f:
        hdf5_psr = f.create_group(psr.name)
        hdf5_psr.create_dataset('toas', psr.toas.shape, psr.toas.dtype, data=psr.toas)
        hdf5_psr.create_dataset('toaerrs', psr.toaerrs.shape,psr.toaerrs.dtype, data=psr.toaerrs)
        hdf5_psr.create_dataset('phi', (1,), float, data=psr.phi)
        hdf5_psr.create_dataset('theta', (1,), float, data=psr.theta)
        hdf5_psr.create_dataset('designmatrix', psr.designmatrix.shape, psr.designmatrix.dtype, data=psr.designmatrix, compression="gzip", compression_opts=compress_val)
        hdf5_psr.create_dataset('G', psr.G.shape, psr.G.dtype, data=psr.G, compression="gzip", compression_opts=compress_val)
        hdf5_psr.create_dataset('K_inv', psr.K_inv.shape, psr.K_inv.dtype, data=psr.K_inv, compression="gzip", compression_opts=compress_val)
        hdf5_psr.create_dataset('pdist', (2,), float, data=psr.pdist)
        f.flush()
        print(f'hasasia pulsar {psr.name} successfully saved to HDF5', end='\r')

def hsen_pulsar_creation(pseudo:PseudoPulsar, hsen_dir:str):
    """_summary_: create hasasia pulsar object using original method

    Args:
        pseudo (PseudoPulsar): PseudoPulsar object
        hsen_dir (str): directory for storing hasasia pulsar object
    """
    #adding red noise covariance matrix responsible for the gravitational wave background based on the selected frequencies
    gwb = hsen.red_noise_powerlaw(A=A_gw, gamma=gam_gw, freqs=freqs_gwb)
    pseudo.N += hsen.corr_from_psd(freqs=freqs_gwb, psd=gwb,
                                toas=pseudo.toas)

    #if instrisic red noise parameters for individual pulsars exist, then add red noise covarariance matrix to it
    if pseudo.name in rn_psrs.keys():
        Amp, gam = rn_psrs[pseudo.name]
        plaw = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs_rn)  #was +=
        pseudo.N += hsen.corr_from_psd(freqs=freqs_rn, psd=plaw,
                                toas=pseudo.toas)   
    #creating hasasia pulsar tobject 
    psr = hsen.Pulsar(toas=pseudo.toas,
                        toaerrs=pseudo.toaerrs,
                        phi=pseudo.phi,theta=pseudo.theta, 
                        N=pseudo.N, designmatrix=pseudo.Mmat, pdist=pseudo.pdist)
    #setting name of hasasia pulsar
    psr.name = pseudo.name
    #calling (GCG^T)^-1 to be computed 
    _ = psr.K_inv
    hsen_pulsar_entry(psr, hsen_dir)

def hsen_pulsar_rrf_creation(pseudo: PseudoPulsar, hsen_dir_rrf:str):
    """_summary_: create hasasia pulsar object using rank-reduced method

    Args:
        pseudo (PseudoPulsar): PseudoPulsar object
        hsen_dir_rrf (str): directory for storing hasasia pulsar object
    """
    psr = hsen.Pulsar(toas=pseudo.toas,
                                    toaerrs=pseudo.toaerrs,
                                    phi=pseudo.phi,theta=pseudo.theta, 
                                    N=pseudo.N, designmatrix=pseudo.Mmat, pdist=pseudo.pdist)
    psr.name = pseudo.name
    _ = psr.K_inv
    hsen_pulsar_entry(psr, hsen_dir_rrf)

def hsen_pulsr_hdf5_entire(f:str, names_list:list, hsen_dir:str):
    """_summary_: Function that goes through entire process of creating fake hasasia pulsar object, create hasasia pulsar object, and 
    saves attributes to hdf5 file. Primary use of this function is to be not executed if already saved. This function is for the original
    method

    Args:
        f (str): enterprise pulsar hdf5 directory
        names_list (list): list of pulsar names
        hsen_dir (str): pulsar hdf5 directory that will used to save the required attributes
    """
    for name in names_list:
        psr = f[name]
        pseudo = PseudoPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                        theta = psr['theta'][:][0], pdist=psr['pdist'][:], N=psr['N'][:])
        pseudo.name = name
        pseudo.Mmat= psr['designmatrix'][:]
        hsen_pulsar_creation(pseudo, hsen_dir)
        del pseudo

def hsen_rrf_pulsar_hdf5_entire(f:str, names_list:list, hsen_dir_rrf:str):
    """_summary_: Function that goes through entire process of creating fake hasasia pulsar object, create hasasia pulsar object, and 
    saves attributes to hdf5 file. Primary use of this function is to be not executed if already saved. This function is for the 
    rank-reduced method

    Args:
        f (str): enterprise pulsar hdf5 directory
        names_list (list): list of pulsar names
        hsen_dir_rrf (str): pulsar hdf5 directory that will used to save the required attributes
    """
    for name in names_list:
        psr = f[name]
        pseudo = PseudoPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                        theta = psr['theta'][:][0], pdist=psr['pdist'][:], N=psr['N'][:])
        pseudo.name = name
        pseudo.Mmat= psr['designmatrix'][:]
        hsen_pulsar_rrf_creation(pseudo, hsen_dir_rrf)
        del pseudo

def hsen_spectrum_creation(pseudo:PseudoSpectraPulsar)->hsen.Spectrum:
    """_summary_: Creates Spectrum object using the original method

    Args:
        pseudo (PseudoSpectraPulsar): fake spectrum pulsar that contains all needed attributes

    Returns:
        hsen.Spectrum: spectrum object
    """
    spec_psr = hsen.Spectrum(pseudo, freqs=freqs)
    spec_psr.name = pseudo.name
    #Calling computation of NcalInv, due to its high computational cost
    _ = spec_psr.NcalInv
    print(f'{spec_psr.name} NcalInv Computed\n')
    return spec_psr

def hsen_spectrum_creation_rrf(pseudo:PseudoSpectraPulsar)-> hsen.RRF_Spectrum:
    """_summary_: Creates Spectrum object using the rank-reduced method

    Args:
        pseudo (PseudoSpectraPulsar): fake spectrum pulsar that contains all needed attributes

    Returns:
        hsen.RRF_Spectrum: spectrum object
    """
    if pseudo.name in rn_psrs.keys():
        Amp, gam = rn_psrs[pseudo.name]
        #creates spectrum pulsar based on both instrinsic red noise and gravitational wave background
        spec_psr = hsen.RRF_Spectrum(psr=pseudo, Tspan=Tspan, freqs_gw=freqs_gwb,amp_gw=A_gw, gamma_gw=gam_gw,
                                     freqs_rn=freqs_rn, amp = Amp, gamma = gam, freqs=freqs)
    else:
        #creates spectrum pulsar just based on gravitational wave background
        spec_psr = hsen.RRF_Spectrum(psr=pseudo, Tspan=Tspan, freqs_gw=freqs_gwb,amp_gw=A_gw, gamma_gw=gam_gw,
                                     freqs_rn=freqs_rn, freqs=freqs)
        
    spec_psr.name = pseudo.name

    _ = spec_psr.NcalInv
    print(f'RRF {spec_psr.name} NcalInv Computed\n')
    return spec_psr

def yr_11_data():
    """Creates enterprise pulsars from the 11 yr dataset from parameter and timing files.

    The quantities that are being returned within this function will be attributes used to write 
    enterprise pulsars onto HDF5 file for 

    Returns:
        - psr_list (list): List of pulsars names
        - enterprise_Psrs (list): List of enterprise pulsars 
        - noise (dict): Noise parameters including fe/be of WN and RN.
        - rn_psrs (dict): RN parameters where key is name of pulsar and value is list where 0th 
        index is spectral amplitude and 1st index is spectral index
        - Tspan: Total timespan of the PTA
        - enterprise_dir: specific directory name used for enterprise HDF5 file
    """
    #File Paths
    pardir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/'
    timdir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/'
    noise_dir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master'
    noise_dir += '/nano11y_data/noisefiles/'
    psr_list_dir = '/home/gourliek/Nanograv/11yr_stochastic_analysis-master/psrlist.txt'

    #organizes files into alphabetical order
    pars = sorted(glob.glob(pardir+'*.par'))
    tims = sorted(glob.glob(timdir+'*.tim'))
    noise_files = sorted(glob.glob(noise_dir+'*.json'))

    #saving pulsar names as a list
    with open(psr_list_dir, 'r') as psr_list_file:
        psr_list = []
        for line in psr_list_file:
            new_line = line.strip("\n")
            psr_list.append(new_line)

    #filtering par and tim files to make sure they only include names found in pulsar list
    pars = [f for f in pars if get_psrname(f) in psr_list]
    tims = [f for f in tims if get_psrname(f) in psr_list]
    noise_files = [f for f in noise_files if get_psrname(f) in psr_list]
    
    if len(pars) == len(tims) and len(tims) == len(noise_files):
        pass
    else:
        print("ERROR. Filteration of tim and par files performed incorrectly")
        exit()

    noise = {}
    for nf in noise_files:
        with open(nf,'r') as fin:
            noise.update(json.load(fin))

    rn_psrs = {'B1855+09':[10**-13.7707, 3.6081],      #
           'B1937+21':[10**-13.2393, 2.46521],
           'J0030+0451':[10**-14.0649, 4.15366],
           'J0613-0200':[10**-13.1403, 1.24571],
           'J1012+5307':[10**-12.6833, 0.975424],
           'J1643-1224':[10**-12.245, 1.32361],
           'J1713+0747':[10**-14.3746, 3.06793],
           'J1747-4036':[10**-12.2165, 1.40842],
           'J1903+0327':[10**-12.2461, 2.16108],
           'J1909-3744':[10**-13.9429, 2.38219],
           'J2145-0750':[10**-12.6893, 1.32307],
           }

    edir = '/11_yr_enterprise_pulsars.hdf5'
    ephem = 'DE436'
    return pars, tims, noise, rn_psrs, edir, ephem

def yr_12_data():
    """Creates enterprise pulsars from the 12.5 yr dataset from parameter and timing files.

    The quantities that are being returned within this function will be attributes used to write 
    enterprise pulsars onto HDF5 file for 

    Returns:
        - psr_list (list): List of pulsars names
        - enterprise_Psrs (list): List of enterprise pulsars 
        - noise (dict): Noise parameters including fe/be of WN and RN.
        - rn_psrs (dict): RN parameters where key is name of pulsar and value is list where 0th 
        index is spectral amplitude and 1st index is spectral index
        - Tspan: Total timespan of the PTA
        - enterprise_dir: specific directory name used for enterprise HDF5 file
    """

    data_dir = r'/home/gourliek/Nanograv/12p5yr_stochastic_analysis-master/data/'
    par_dir = data_dir + r'par/'
    tim_dir = data_dir + r'tim/'
    noise_file = data_dir + r'channelized_12p5yr_v3_full_noisedict.json' 
  
    #sorting parameter and timing files
    parfiles = sorted(glob.glob(par_dir+'*.par'))
    timfiles = sorted(glob.glob(tim_dir+'*.tim'))

    #getting names of pulsars from timing files
    par_psr_names = []
    for file in parfiles:
        par_psr_names.append(get_psrname(file))

    #getting names of pulsars from parameter files
    tim_psr_names = []
    for file in timfiles:
        tim_psr_names.append(get_psrname(file))

    #grabbing intersection of names
    psr_list= [item for item in tim_psr_names if item in par_psr_names]
    
    pars_v1 = [f for f in parfiles if get_psrname(f) in psr_list]

     # ...filtering out the tempo parfile...
    pars = [x for x in pars_v1 if 'J1713+0747_NANOGrav_12yv3.gls.par' not in x]
    tims = [f for f in timfiles if get_psrname(f) in psr_list]

    noise = {}
    with open(noise_file, 'r') as fp:
        noise.update(json.load(fp))

    #initialize dictionary list with placeholders where parameters for rn will be held
    rn_psrs = {}
    for name in psr_list:
        amp_key = name + '_red_noise_log10_A'
        gamma_key = name + '_red_noise_gamma'
        for key in noise:
            if key == amp_key or key == gamma_key:
                rn_psrs[name] = ['x','x']
    
    #place proper entries
    for name in psr_list:
        amp_key = name + '_red_noise_log10_A'
        gamma_key = name + '_red_noise_gamma'
        for key in noise:
            if key == amp_key:
                rn_psrs[name][0] = 10**noise[amp_key]  #because parameter is log_10()
            elif key == gamma_key:
                rn_psrs[name][1] = noise[gamma_key]

    edir = '/12_yr_enterprise_pulsars.hdf5'
    ephem = 'DE438'
    
    return pars, tims, noise, rn_psrs, edir, ephem



def yr_15_data():
    data_dir =  os.path.expanduser('~/Research/Nanograv/NANOGrav15yr_PulsarTiming_v2.0.0/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0/narrowband/')
    par_dir = data_dir + r'par/'
    tim_dir = data_dir + r'tim/'
    noise_file = data_dir+r'15yr_wn_dict.json'
    jeremy_psrs =["B1855+09","B1937+21","B1953+29","J0023+0923","J0030+0451","J0340+4130","J0406+3039","J0437-4715","J0509+0856",
                        "J0557+1551","J0605+3757","J0610-2100","J0613-0200","J0636+5128","J0645+5158","J0709+0458","J0740+6620",
                        "J0931-1902","J1012+5307","J1012-4235","J1022+1001","J1024-0719","J1125+7819","J1312+0051","J1453+1902",
                        "J1455-3330","J1600-3053","J1614-2230","J1630+3734","J1640+2224","J1643-1224","J1705-1903","J1713+0747",
                        "J1719-1438","J1730-2304","J1738+0333","J1741+1351","J1744-1134","J1745+1017","J1747-4036","J1751-2857",
                        "J1802-2124","J1811-2405","J1832-0836","J1843-1113","J1853+1303","J1903+0327","J1909-3744","J1910+1256",
                        "J1911+1347","J1918-0642","J1923+2515","J1944+0907","J1946+3417","J2010-1323","J2017+0603","J2033+1734",
                        "J2043+1711","J2124-3358","J2145-0750","J2214+3000","J2229+2643","J2234+0611","J2234+0944","J2302+4442",
                        "J2317+1439","J2322+2057"]

    #sorting parameter and timing files
    parfiles = sorted(glob.glob(par_dir+'*.par'))
    timfiles = sorted(glob.glob(tim_dir+'*.tim'))
   
    filter_parfiles = []
    for file in parfiles:
        if 'ao' in file:
            continue
        if 'gbt' in file:
            continue
        filter_parfiles.append(file)

    filter_timfiles = []
    for file in timfiles:
        if 'ao' in file:
            continue
        if 'gbt' in file:
            continue
        filter_timfiles.append(file)
    
    del parfiles, timfiles

    par_psr_names = []
    for file in filter_parfiles:
        par_psr_names.append(get_psrname(file))
    

    tim_psr_names = []
    for file in filter_timfiles:
        tim_psr_names.append(get_psrname(file))

    psr_list= [item for item in tim_psr_names if item in par_psr_names]
    
    exclude_psr = next(iter(set(psr_list) - set(jeremy_psrs)))
    
    pars =[]
    for file in filter_parfiles:
        if get_psrname(file) == exclude_psr:
            continue
        pars.append(file)

    tims = []
    for file in filter_timfiles:
        if get_psrname(file) == exclude_psr:
            continue
        tims.append(file)
    
    del filter_parfiles, filter_timfiles

    if len(pars) != 67 or len(tims) !=67:
        exit()

    noise = {}
    with open(noise_file, 'r') as fp:
        noise.update(json.load(fp))        
    
    rn_psrs = {}
    for name in psr_list:
        amp_key = name + '_red_noise_log10_A'
        gamma_key = name + '_red_noise_gamma'
        for key in noise:
            if key == amp_key or key == gamma_key:
                rn_psrs[name] = ['x','x']
    
    #place proper entries
    for name in jeremy_psrs:
        amp_key = name + '_red_noise_log10_A'
        gamma_key = name + '_red_noise_gamma'
        for key in noise:
            if key == amp_key:
                rn_psrs[name][0] = 10**noise[amp_key]  #because parameter is log_10()
            elif key == gamma_key:
                rn_psrs[name][1] = noise[gamma_key]

    edir = '/15_yr_enterprise_pulsars.hdf5'
    ephem = 'DE440'

    return pars, tims, noise, rn_psrs, edir, ephem

def chains_puller(yr:float):
    """_summary_: Reads generated chains from the 12.5 yr data, varying spectral index, 30 frequencies, from the DE438 ephemeris.

    Resource: https://nanograv.org/science/data/125-year-stochastic-gravitational-wave-background-search

    Args:
        num (int): number of random samples

    Returns:
        Two numpy arrays containing the random samples
    """
    if yr == 12.5:
        chain_path = os.path.expanduser('~/Nanograv/12p5yr_varying_sp_ind_30freqs/12p5yr_DE438_model2a_cRN30freq_gammaVary_chain.hdf5')
        with h5py.File(chain_path, 'r') as chainf:
            params = chainf['params'][:]
            samples = np.array(chainf['samples'][:])
            
        list_params = [item.decode('utf-8') for item in params]

        lnpost = samples[:,-4]
        lnlike = samples[:,-3]
        chain_accept = samples[:,-2]
        pt_chain_accept = samples[:,-1]

        lnpost_max = np.argmax(lnpost)
        
        gw_log10_A_ind = list_params.index('gw_log10_A')
        gw_gamma_ind = list_params.index('gw_gamma')

        gw_log10_A_samples = samples[30000:,gw_log10_A_ind]
        gw_gamma_samples = samples[30000:,gw_gamma_ind]
    
        
        return gw_log10_A_samples[lnpost_max], gw_gamma_samples[lnpost_max]
    
    elif yr == 15:
        chain_path = os.path.expanduser('~/Research/Nanograv/NANOGrav15yr_PulsarTiming_v2.0.0/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0/curn_gamma_14f_noBE/data/taylor_group/nihan_pol/15yr_v1p1/pint/model_2a_vg_noBE_14f/')
        chain_par = chain_path+r'pars.txt'
        chain_chains = chain_path+r'chain_1.0.txt'

        list_params = []
        with open(chain_par, 'r') as file:
            for line in file:
                line = line.strip('\n')
                list_params.append(line)

        gw_log10_A_ind = list_params.index('gw_crn_log10_A')
        gw_gamma_ind = list_params.index('gw_crn_gamma')

        with open(chain_chains, 'r') as file:
            samples = np.loadtxt(file)

        lnpost = samples[:,-4]
        lnlike = samples[:,-3]
        chain_accept = samples[:,-2]
        pt_chain_accept = samples[:,-1]

        lnpost_max = np.argmax(lnpost)

        gw_log10_A_samples = samples[30000:,gw_log10_A_ind]
        gw_gamma_samples = samples[30000:,gw_gamma_ind]
        
        return gw_log10_A_samples[lnpost_max], gw_gamma_samples[lnpost_max]


def yr_15_free_spec(names_list, num):
    chain_path = os.path.expanduser('~/Research/Nanograv/NANOGrav15yr_PulsarTiming_v2.0.0/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0/30fCP_30fiRN_2A.core')
    with h5py.File(chain_path, 'r') as chainf:
        #jump_fractions = chainf['jump_fractions']
        #meta_data = chainf['metadata']
        #priors = chainf['priors']
        params = chainf['params'][:]
        freqs_rn = np.array(chainf['rn_freqs'][:])
        #cov = chainf['cov'][:]
        chains = np.array(chainf['chain'][:][30000:,:])

    
    list_params = [item.decode('utf-8') for item in params]
    lnpost = chains[:,-4]
    lnpost_max = np.argmax(lnpost)

    rand_samples_ind = np.random.choice(a=chains.shape[0], size=num, replace=False)
    rn_params_log10_A = {}
    rn_params_gam = {}
    rn_max_log10_A = {}
    rn_max_gam = {}
    gwb_free_specs = []
    gwb_free_specs_max = []
        
    for name in names_list:
        print(name)
        log10_A_ind = list_params.index(name+'_red_noise_log10_A')
        gamma_ind = list_params.index(name+'_red_noise_gamma')
        log10_A_samples = chains[:,log10_A_ind]
        gamma_samples = chains[:,gamma_ind]
        free_spec_ind = list_params.index('gw_crn_log10_rho_' + name[0])
        free_spec_samples = chains[:,free_spec_ind]

        rn_params_log10_A[name] = log10_A_samples[rand_samples_ind]
        rn_params_gam[name] = gamma_samples[rand_samples_ind]
        rn_max_log10_A[name] = log10_A_samples[lnpost_max]
        rn_max_gam[name] = gamma_samples[lnpost_max]


    for i in range(freqs_rn.size):
        free_spec_ind = list_params.index('gw_crn_log10_rho_' + i)
        free_spec_samples = chains[:,free_spec_ind]
        gwb_free_specs.append(free_spec_samples[rand_samples_ind])
        gwb_free_specs_max.append(free_spec_samples[lnpost_max])

        return freqs_rn, rn_params_log10_A, rn_max_log10_A, rn_params_gam, rn_max_gam, gwb_free_specs, gwb_free_specs_max



if __name__ == '__main__':
    fyr = 1/(365.25*24*3600)   
    #max is 34 for 11yr dataset
    #max is 45 for 12yr dataset
    #max is 67 for 15yr dataset
    kill_count =  67
    max_harm = 70
    #intrinsic red noise harmonic number, 30
    irn_harms = 30
    #gravitational wave backgroun harmonic number, 14
    gwb_harms = 14
    #thinning values, if you choose
    thin_val = 10
    #thin_val = None

    #compress_val=0 means no compression
    compress_val = 9
    names_list = []

    #Realistic PTA datasets
    #pars, tims, noise, rn_psrs, edir, ephem = yr_11_data()
    #pars, tims, noise, rn_psrs, edir, ephem = yr_12_data()
    pars, tims, noise, rn_psrs, edir, ephem = yr_15_data()
    
    if ephem == 'DE436':
        yr = 11
        A_gw = 1.73e-15
        gam_gw = 13./3
    elif ephem == 'DE438':
        yr=12.5
        log10_A_gw, gam_gw = chains_puller(yr)
        A_gw = 10**log10_A_gw
    elif ephem == 'DE440':
        yr = 15
        log10_A_gw, gam_gw = chains_puller(yr)
        A_gw = 10**log10_A_gw

    print(f'A_gw: {A_gw}\tgam_gw: {gam_gw}')
    data_path = os.path.expanduser(f'~/psr_data_{yr}_yr')
    try:
        os.mkdir(data_path)
    
    except FileExistsError:
        print('Pulsar data folder exists.\nLoading data...')
    
    edir = data_path + edir

    pickle_dir = os.path.expanduser(data_path+f'/{yr}_enterprise_pulsars.pkl')
    if not os.path.isfile(pickle_dir):
        ePsrs = enterprise_creation(pars, tims, ephem)
        enterprise_pickle(ePsrs, pickle_dir)
        del ePsrs

    pkl_psrs = pickle_enterprise(pickle_dir)
    if not os.path.isfile(edir):
        enterprise_hdf5(pkl_psrs, noise, yr, edir)
    del pkl_psrs

    #reading hdf5 file containing enterprise.pulsar attributes
    with h5py.File(edir, 'r') as f:
        #reading Tspan and creation of frequencies to observe
        Tspan = f['Tspan'][:][0]
        freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(max_harm/Tspan),400)
        freqs_rn = np.linspace(1/Tspan, irn_harms/Tspan, irn_harms)
        freqs_gwb = np.linspace(1/Tspan, gwb_harms/Tspan, gwb_harms)

        #reading names encoded as bytes, and re-converting them to strings, and deleting byte names
        names = f['names'][:]
        for i in range(len(names)):
            names_list.append(names[i].decode('utf-8'))
        del names

        #Original Method for creation of hasasia pulsars, and saving them to hdf5 file
        hsen_dir = os.path.expanduser(f'{data_path}/hsen_psrs.hdf5')
        if not os.path.isfile(hsen_dir):
            hsen_pulsr_hdf5_entire(f, names_list, hsen_dir)
            
            
        #Rank-Reduced Method for creation of hasasia pulsars, and saving them to hdf5 file
        hsen_dir_rrf = os.path.expanduser(f'{data_path}/hsen_psrs_rrf.hdf5')
        if not os.path.isfile(hsen_dir_rrf):
            hsen_rrf_pulsar_hdf5_entire(f, names_list, hsen_dir_rrf)


        hc_dir = os.path.expanduser(data_path+f'/{yr}_hc_data.npz')
        if not os.path.isfile(hc_dir):       
            #reading hdf5 file containing hasasia pulsar attributes from original method to create list of spectrum objects
            with h5py.File(hsen_dir,'r') as hsenf:
                specs = []
                for name in names_list:
                    psr = hsenf[name]
                    pseudo = PseudoSpectraPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                                            theta = psr['theta'][:][0], pdist=psr['pdist'][:], K_inv=psr['K_inv'][:], G=psr['G'][:],
                                            designmatrix=psr['designmatrix'])
                    del psr
                    pseudo.name = name
                    spec = hsen_spectrum_creation(pseudo)
                    del pseudo
                    gc.collect()
                    specs.append(spec)

            #creation of sensitivity curves original method
            sc = hsen.GWBSensitivityCurve(specs)
            dsc = hsen.DeterSensitivityCurve(specs, A_GWB=A_gw)
            del specs
            sc_hc = sc.h_c
            sc_freqs = sc.freqs
            dsc_hc = dsc.h_c
            dsc_freqs = dsc.freqs
            del sc, dsc
                    
            #reading hdf5 file containing hasasia pulsar attributes from rank-reduced method to create list of spectrum objects     
            with h5py.File(hsen_dir_rrf,'r') as hsenfrrf:
                specs_rrf = []
                for name in names_list:
                    psr = hsenfrrf[name]
                    pseudo = PseudoSpectraPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                                            theta = psr['theta'][:][0], pdist=psr['pdist'][:], K_inv=psr['K_inv'][:], G=psr['G'][:],
                                            designmatrix=psr['designmatrix'])
                    del psr
                    gc.collect()
                    pseudo.name = name
                    spec_psr_rrf = hsen_spectrum_creation_rrf(pseudo)
                    del pseudo
                    delattr(spec_psr_rrf, 'K_inv')
                    delattr(spec_psr_rrf, 'G')
                    delattr(spec_psr_rrf, 'J')
                    delattr(spec_psr_rrf, 'Z')
                    delattr(spec_psr_rrf, 'Cirn')
                    delattr(spec_psr_rrf, 'Cgw')
                    gc.collect()
                    specs_rrf.append(spec_psr_rrf)
                    
            
            #creation of sensitivity curves rank-reduced method
            rrf_sc = hsen.GWBSensitivityCurve(specs_rrf)
            rrf_dsc = hsen.DeterSensitivityCurve(specs_rrf, A_GWB=A_gw)
            del specs_rrf
            rrf_sc_hc = rrf_sc.h_c
            rrf_sc_freqs = rrf_sc.freqs
            rrf_dsc_hc = rrf_dsc.h_c
            rrf_dsc_freqs = rrf_dsc.freqs
            del rrf_sc, rrf_dsc

            np.savez(hc_dir, rrf_sc_hc=rrf_sc_hc, rrf_dsc_hc=rrf_dsc_hc, sc_hc=sc_hc, dsc_hc = dsc_hc)
            del rrf_sc_hc, rrf_dsc_hc, sc_hc, dsc_hc

        hc_data_loaded = np.load(hc_dir)

        # Access arrays by their names
        rrf_sc_hc_loaded = hc_data_loaded['rrf_sc_hc']
        rrf_dsc_hc_loaded = hc_data_loaded['rrf_dsc_hc']
        sc_hc_loaded = hc_data_loaded['sc_hc']
        dsc_hc_loaded = hc_data_loaded['dsc_hc']


        a_gw_base10, a_gw_exp = f"{A_gw:.2e}".split("e")
        a_gw_base10 = float(a_gw_base10)  # Convert to float
        a_gw_exp = int(a_gw_exp)    # Convert exponent to integer
        gam_gw = float(gam_gw)

        #plotting sensitivity curves
        #plt.loglog(freqs, sc_hc_loaded, label='Stoch', c='red')
        #plt.loglog(freqs, dsc_hc_loaded, label='Det', c='blue')
        plt.axvline(x=1/Tspan, label=r'$\frac{1}{\mathrm{Tspan}}$', c='peru', linestyle='--')
        plt.axvline(x=14/Tspan, label=fr'$\frac{{{14}}}{{\mathrm{{Tspan}}}}$', c='teal', linestyle='--')
        plt.axvline(x=30/Tspan, label=fr'$\frac{{{30}}}{{\mathrm{{Tspan}}}}$', c='fuchsia', linestyle='--')
        plt.loglog(freqs, sc_hc_loaded, label='Wiener-Khinchin Stoch', c='red')
        plt.loglog(freqs, dsc_hc_loaded, label='Wiener-Khinchin Det', c='blue')
        plt.loglog(freqs, rrf_sc_hc_loaded, label='Rank-Reduced Stoch', c='orange', linestyle='--')
        plt.loglog(freqs, rrf_dsc_hc_loaded, label='Rank-Reduced Det', c='cyan', linestyle='--')
        plt.plot([], [], label=f'$A_{{gwb}}: {a_gw_base10} \\times 10^{{{a_gw_exp}}}$', color='none')
        plt.plot([], [],label=f'$\gamma_{{gwb}}: {round(gam_gw, 2)}$', color='none')
        plt.ylabel('Characteristic Strain, $h_c$')
        plt.xlabel('Frequency (Hz)')
        plt.xlim(np.min(freqs), np.max(freqs))
        plt.title(f'NG{yr}YR Max Posterior Sensitivity Curve')
        plt.grid(which='both', alpha=0.2)
        plt.legend(loc='upper left', fontsize='small')
        plt.savefig(data_path+'/sc_h_c.svg')
        plt.savefig(data_path+'/sc_h_c_trans.svg', transparent=True)
        plt.show()
        plt.close()





    