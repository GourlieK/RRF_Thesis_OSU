#Kyle Gourlie
#8/13/2024
#library imports
import os, gc, h5py, gc, glob, json, pickle, time
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib as mpl
import healpy as hp
import astropy.units as u
import astropy.constants as c
from enterprise.pulsar import Pulsar as ePulsar

#hasasia imports that are modified so they can be benchmarked
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

def enterprise_hdf5(ePsrs:list, noise:dict, yr:float, edir:str, thin_val):
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
        hdf5_psr.create_dataset('designmatrix', psr.designmatrix.shape, psr.designmatrix.dtype, data=psr.designmatrix)
        hdf5_psr.create_dataset('G', psr.G.shape, psr.G.dtype, data=psr.G)
        hdf5_psr.create_dataset('K_inv', psr.K_inv.shape, psr.K_inv.dtype, data=psr.K_inv)
        hdf5_psr.create_dataset('pdist', (2,), float, data=psr.pdist)
        f.flush()
        print(f'hasasia pulsar {psr.name} successfully saved to HDF5', end='\r')

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
    return spec_psr


def hsen_spectrum_creation_rrf(pseudo:PseudoSpectraPulsar, gam_gw, A_gw, A_irn=None, gam_irn=None, rn_psrs=None)-> hsen.RRF_Spectrum:
    """_summary_: Creates Spectrum object using the rank-reduced method

    Args:
        pseudo (PseudoSpectraPulsar): fake spectrum pulsar that contains all needed attributes

    Returns:
        hsen.RRF_Spectrum: spectrum object
    """
    if A_irn is None or gam_irn is None:
        if pseudo.name in rn_psrs.keys():
            Amp, gam = rn_psrs[pseudo.name]
            #creates spectrum pulsar based on both instrinsic red noise and gravitational wave background
            spec_psr = hsen.RRF_Spectrum(psr=pseudo, Tspan=Tspan, freqs_gw=freqs_gwb,amp_gw=A_gw, gamma_gw=gam_gw,
                                        freqs_rn=freqs_rn, amp = Amp, gamma = gam, freqs=freqs)
        else:
            #creates spectrum pulsar just based on gravitational wave background
            spec_psr = hsen.RRF_Spectrum(psr=pseudo, Tspan=Tspan, freqs_gw=freqs_gwb,amp_gw=A_gw, gamma_gw=gam_gw,
                                        freqs_rn=freqs_rn, freqs=freqs)
    if rn_psrs is None:
        spec_psr = hsen.RRF_Spectrum(psr=pseudo, Tspan=Tspan, freqs_gw=freqs_gwb,amp_gw=A_gw, gamma_gw=gam_gw,
                                        freqs_rn=freqs_rn, amp = A_irn, gamma = gam_irn, freqs=freqs) 
    spec_psr.name = pseudo.name
    _ = spec_psr.NcalInv
    return spec_psr







def yr_11_data():
    #File Paths
    pardir = os.path.expanduser('~/Research/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/')
    timdir = os.path.expanduser('~/Research/Nanograv/11yr_stochastic_analysis-master/nano11y_data/partim/')
    noise_dir = os.path.expanduser('~/Research/Nanograv/11yr_stochastic_analysis-master')
    noise_dir += '/nano11y_data/noisefiles/'
    psr_list_dir = os.path.expanduser('~/Research/Nanograv/11yr_stochastic_analysis-master/psrlist.txt')

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
    data_dir = os.path.expanduser('~/Research/Nanograv/12p5yr_stochastic_analysis-master/data/')
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
    data_dir = os.path.expanduser('~/Research/Nanograv/NANOGrav15yr_PulsarTiming_v2.0.0/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0/narrowband/')
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


def chains_puller(num: int, names_list:list, yr:float):
    """_summary_: Reads generated chains from the 12.5 yr data, varying spectral index, 30 frequencies, from the DE438 ephemeris.

    Resource: https://nanograv.org/science/data/125-year-stochastic-gravitational-wave-background-search

    Args:
        num (int): number of random samples
        names_list (list): List of pulsar names

    Returns:
        Two dictionaries containing instrinsic red noise parameters
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

        lnpost_max_prob_ind = np.argmax(lnpost)
        gw_log10_A_ind = list_params.index('gw_log10_A')
        gw_gamma_ind = list_params.index('gw_gamma')

        gw_log10_A_samples = samples[30000:,gw_log10_A_ind]
        gw_gamma_samples = samples[30000:,gw_gamma_ind]

        #generating random indices to draw samples from
        rand_samples_ind = np.random.choice(a=gw_log10_A_samples.shape[0], size=num, replace=False)

        gw_log10_A_max_lnpost = samples[lnpost_max_prob_ind, gw_log10_A_ind]
        gw_gamma_max_lnpost = samples[lnpost_max_prob_ind, gw_gamma_ind]

        gw_log10_A_rand_samples = gw_log10_A_samples[rand_samples_ind]
        gw_gamma_rand_samples = gw_gamma_samples[rand_samples_ind]
        

        irn_params_log10_A = {}
        irn_params_gam = {}

        for name in names_list:
            log10_A_ind = list_params.index(name+'_red_noise_log10_A')
            gamma_ind = list_params.index(name+'_red_noise_gamma')
            log10_A_samples = samples[30000:,log10_A_ind]
            gamma_samples = samples[30000:,gamma_ind]
            irn_params_log10_A[name] = log10_A_samples[rand_samples_ind]
            irn_params_gam[name] = gamma_samples[rand_samples_ind]


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

        gw_log10_A_max_lnpost = gw_log10_A_samples[lnpost_max]
        gw_gamma_max_lnpost = gw_gamma_samples[lnpost_max]
        
        irn_params_log10_A = {}
        irn_params_gam = {}

        log10_A_ind = list_params.index(names_list[0]+'_red_noise_log10_A')
        log10_A_samples = samples[30000:,log10_A_ind]
        rand_samples_ind = np.random.choice(a=log10_A_samples.shape[0], size=num, replace=False)

        for name in names_list:
            log10_A_ind = list_params.index(name+'_red_noise_log10_A')
            gamma_ind = list_params.index(name+'_red_noise_gamma')
            log10_A_samples = samples[30000:,log10_A_ind]
            gamma_samples = samples[30000:,gamma_ind]
            irn_params_log10_A[name] = log10_A_samples[rand_samples_ind]
            irn_params_gam[name] = gamma_samples[rand_samples_ind]

        gw_log10_A_rand_samples =  gw_log10_A_samples[rand_samples_ind] 
        gw_gamma_rand_samples = gw_gamma_samples[rand_samples_ind] 

    return irn_params_log10_A, irn_params_gam, gw_log10_A_rand_samples, gw_gamma_rand_samples, gw_log10_A_max_lnpost, gw_gamma_max_lnpost



def lnpost_max_run(gw_log10_A_max, gw_gamma_max, rn_psrs_max:dict):
    with h5py.File(hsen_dir_rrf,'r') as hsenfrrf:
        specs_rrf = []
        for name in names_list:
            psr = hsenfrrf[name]
            pseudo = PseudoSpectraPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                                    theta = psr['theta'][:][0], pdist=psr['pdist'][:], K_inv=psr['K_inv'][:], G=psr['G'][:],
                                    designmatrix=psr['designmatrix'])
            pseudo.name = name
            spec_psr_rrf = hsen_spectrum_creation_rrf(pseudo, gw_gamma_max, 10**gw_log10_A_max, rn_psrs=rn_psrs_max)
            specs_rrf.append(spec_psr_rrf)
                
                #creation of sensitivity curves rank-reduced method
    rrf_sc_maxpost = hsen.GWBSensitivityCurve(specs_rrf)
    rrf_dsc_maxpost = hsen.DeterSensitivityCurve(specs_rrf)
    del specs_rrf
    return rrf_sc_maxpost.h_c, rrf_dsc_maxpost.h_c


def save_h_c(data_path, hc_dsc, hc_sc, a_gw, gam_gw, a_irn, gam_irn, freqs, batch_num: int): 
    with h5py.File(data_path, 'a') as f:
        hdf5 = f.create_group(f'batch_{batch_num}')
        hdf5.create_dataset('A_irn', (1,), float, data=a_irn)
        hdf5.create_dataset('gam_irn', (1,), float, data=gam_irn)
        hdf5.create_dataset('A_gwb', (1,), float, data=a_gw)
        hdf5.create_dataset('gam_gwb', (1,), float, data=gam_gw)
        hdf5.create_dataset('hc_dsc', hc_dsc.shape, hc_dsc.dtype, data=hc_dsc)
        hdf5.create_dataset('hc_sc', hc_sc.shape, hc_sc.dtype, data=hc_sc)
        hdf5.create_dataset('freqs', freqs.shape, freqs.dtype, data=freqs)
        f.flush()

def sens_gen(irn_log10_As, irn_gams, gw_log10_As, gw_gammas):
    
    for i in range(num_chains):
        gam_gw = gw_gammas[i]
        log_10_A_gw = gw_log10_As[i]
        print(f'GWB Spectral Index:{gam_gw}')
        print(f'GWB Spectral Amplitude:{10**log_10_A_gw}\n')
        
        #reading hdf5 file containing hasasia pulsar attributes from rank-reduced method to create list of spectrum objects     
        with h5py.File(hsen_dir_rrf,'r') as hsenfrrf:
            timestart = time.time()
            specs_rrf = []
            for name in names_list:
                psr = hsenfrrf[name]
                pseudo = PseudoSpectraPulsar(toas=psr['toas'][:], toaerrs=psr['toaerrs'][:], phi = psr['phi'][:][0],
                                        theta = psr['theta'][:][0], pdist=psr['pdist'][:], K_inv=psr['K_inv'][:], G=psr['G'][:],
                                        designmatrix=psr['designmatrix'])
                del psr
                gc.collect() 
                pseudo.name = name
                psr_log10_A = irn_log10_As[pseudo.name][i]
                psr_gam = irn_gams[pseudo.name][i]
                print(f'IRN Spectral Index:{psr_gam}')
                print(f'IRN Spectral Amplitude:{10**psr_log10_A}')
                spec_psr_rrf = hsen_spectrum_creation_rrf(pseudo, gam_gw=gam_gw, A_gw= 10**log_10_A_gw, A_irn=10**psr_log10_A, gam_irn=psr_gam)
                del pseudo
                gc.collect()
                specs_rrf.append(spec_psr_rrf)

        timend = time.time()
        time_diff = timend - timestart
        mem_inc_spec.write(f'{time_diff}\n')     
        mem_inc_spec.flush()  
        #creation of sensitivity curves rank-reduced method
        rrf_sc = hsen.GWBSensitivityCurve(specs_rrf)
        rrf_dsc = hsen.DeterSensitivityCurve(specs_rrf)
        del specs_rrf
        save_h_c(data_path_hc, rrf_dsc.h_c, rrf_sc.h_c,
                    10**log_10_A_gw, gam_gw, 10**psr_log10_A, psr_gam, rrf_sc.freqs, i)
        del rrf_sc
        del rrf_dsc
        print()
        print(f'Batch #{i+1} finished!')
        print()


def sigma_grab(sigma:int):
    dz = 0.01
    input = np.arange(-sigma, sigma+dz, dz)
    output = np.exp(-0.5*input**2)
    sigma_val = np.trapz(y=output, x=input, dx=dz)/np.sqrt(2*np.pi)
    lower = .5 - sigma_val/2
    upper = .5 + sigma_val/2
    return lower, upper




if __name__ == '__main__':
    ###################################################
    #max is 34 for 11yr dataset
    #max is 45 for 12yr dataset
    sigma = 2
    kill_count = 67
    num_chains = 10
    #thin_val = 20
    thin_val=None
    compress_val = 9

    #yr used for making WN correlation matrix, specifically when yr=15
    freq_harm_gwb = 14
    freq_harm_irn = 30
    fyr = 1/(365.25*24*3600)

    names_list = []
    #Realistic PTA datasets
    #pars, tims, noise, rn_psrs_partim, edir, ephem = yr_11_data()
    #pars, tims, noise, rn_psrs_partim, edir, ephem = yr_12_data()
    pars, tims, noise, rn_psrs_partim, edir, ephem = yr_15_data()

    if ephem == 'DE436':
        yr = 11
    elif ephem == 'DE438':
        yr=12.5
    elif ephem == 'DE440':
        yr = 15

    data_path = os.path.expanduser(f'~/psr_data_{yr}_yr')
    profile_path = os.path.expanduser(f'~/profile_data_sens_var_{yr}_yr')
    try:
        os.mkdir(data_path)
    
    except FileExistsError:
        print('Pulsar data folder exists.\nLoading data...')
    
    edir = data_path + edir

    os.mkdir(profile_path)
    mem_inc_spec = open(profile_path + '/specs_RRF_mem.txt', 'w')

    pickle_dir = os.path.expanduser(data_path+f'/{yr}_enterprise_pulsars.pkl')
    if not os.path.isfile(pickle_dir):
        ePsrs = enterprise_creation(pars, tims, ephem)
        enterprise_pickle(ePsrs, pickle_dir)
        del ePsrs

    pkl_psrs = pickle_enterprise(pickle_dir)
    if not os.path.isfile(edir):
        enterprise_hdf5(pkl_psrs, noise, yr, edir, thin_val)
        del pkl_psrs

    #if file does not exist, then re-compute it
    if not os.path.isfile(edir):
        ePsrs = enterprise_creation(pars, tims, ephem)
        enterprise_hdf5(ePsrs, noise, yr, edir, thin_val)
        del ePsrs

    #reading hdf5 file containing enterprise.pulsar attributes
    with h5py.File(edir, 'r') as f:
        #reading Tspan and creation of frequencies to observe
        Tspan = f['Tspan'][:][0]

        freqs = np.logspace(np.log10(1/(5*Tspan)),np.log10(2e-7),400)
        freqs_rn = np.linspace(1/Tspan, freq_harm_irn/Tspan, freq_harm_irn)
        freqs_gwb = np.linspace(1/Tspan, freq_harm_gwb/Tspan, freq_harm_gwb)

        #reading names encoded as bytes, and re-converting them to strings, and deleting byte names
        names = f['names'][:]
        for i in range(kill_count):
            names_list.append(names[i].decode('utf-8'))
        del names    
            
        #Rank-Reduced Method for creation of hasasia pulsars, and saving them to hdf5 file
        hsen_dir_rrf = os.path.expanduser(data_path+'/hsen_psrs_rrf.hdf5')
        if not os.path.isfile(hsen_dir_rrf):
            hsen_rrf_pulsar_hdf5_entire(f, names_list, hsen_dir_rrf)

    irn_params_log10_A, irn_params_gam, gw_log10_A_rand_samples, gw_gamma_rand_samples, gw_log10_A_max_lnpost, gw_gamma_max_lnpost = chains_puller(num_chains, names_list, yr)
    data_path_hc = os.path.expanduser(data_path+'/h_c_data.hdf5')
    if not os.path.isfile(data_path_hc):
        sens_gen(irn_params_log10_A, irn_params_gam, gw_log10_A_rand_samples, gw_gamma_rand_samples)

    mem_inc_spec.close()

    rrf_sc_maxpost_h_c, rrf_dsc_maxpost_h_c = lnpost_max_run(gw_log10_A_max_lnpost, gw_gamma_max_lnpost, rn_psrs_partim)
    

    h_sc_matrix = np.zeros((freqs.size, num_chains))
    h_dsc_matrix = np.zeros((freqs.size, num_chains))

    
    #plotting sensitivity curves
    
    for i in range(num_chains):
        with h5py.File(data_path_hc, 'r') as data_f:
            data = data_f['batch_'+str(i)]
            freqs = data['freqs'][:]
            h_sc_matrix[:,i] = data['hc_sc'][:]
            h_dsc_matrix[:,i] = data['hc_dsc'][:]

    h_sc_sigma_low = []
    h_sc_med = []
    h_sc_sigma_high = []

    h_dsc_sigma_low = []
    h_dsc_med = []
    h_dsc_sigma_high = []

    low, high = sigma_grab(sigma=sigma)
    
    
    for i in range(freqs.size):
        h_sc_sigma_low.append(np.quantile(a=h_sc_matrix[i,:], q=low))
        h_sc_med.append(np.median(h_sc_matrix[i,:]))
        h_sc_sigma_high.append(np.quantile(a=h_sc_matrix[i,:], q=high))

        h_dsc_sigma_low.append(np.quantile(a=h_dsc_matrix[i,:], q=low))
        h_dsc_med.append(np.median(h_dsc_matrix[i,:]))
        h_dsc_sigma_high.append(np.quantile(a=h_dsc_matrix[i,:], q=high))


    plt.title(f'NANOGrav {yr}-year Data Set Sensitivity Curve')
    plt.loglog(freqs, h_sc_med, label='Median Stoch', lw=0.2, c='blue')
    plt.loglog(freqs, h_dsc_med, label='Median Det', lw=0.2, c='red')
    

    plt.fill_between(freqs, h_sc_sigma_low, h_sc_sigma_high, color='cyan', label=f'{sigma}$\sigma$ Stoch')
    plt.fill_between(freqs, h_dsc_sigma_low, h_dsc_sigma_high, color='orange', label=f'{sigma}$\sigma$ Det')

    
    plt.tight_layout()
    plt.xlabel('Frequencies, Hz')
    plt.ylabel('Characteristic Strain, $h_c$')
    plt.grid(which='both', alpha=0.2)
    
    plt.axvline(x=1/Tspan, label=r'$\frac{1}{\mathrm{Tspan}}$', c='peru', linestyle='--')
    plt.axvline(x=14/Tspan, label=fr'$\frac{{{14}}}{{\mathrm{{Tspan}}}}$', c='teal', linestyle='--')
    plt.axvline(x=30/Tspan, label=fr'$\frac{{{30}}}{{\mathrm{{Tspan}}}}$', c='fuchsia', linestyle='--')
    plt.legend()
    plt.savefig(data_path+'/sc_h_c_sigma.svg')
    plt.show()
    plt.close()

    plt.title(f'NANOGrav {yr}-year Data Set Sensitivity Curve')
    for i in range(num_chains):
        plt.loglog(freqs, h_sc_matrix[:,i], c='blue', lw=0.2)
        plt.loglog(freqs, h_dsc_matrix[:,i], c='red', lw=0.2)

    plt.loglog(freqs, h_sc_matrix[:,i], label='Stoch', c='blue', lw=0.2)
    plt.loglog(freqs, h_dsc_matrix[:,i], label='Det', c='red', lw=0.2)
    plt.tight_layout()
    plt.axvline(x=1/Tspan, label=r'$\frac{1}{\mathrm{Tspan}}$', c='peru', linestyle='--')
    plt.axvline(x=14/Tspan, label=fr'$\frac{{{14}}}{{\mathrm{{Tspan}}}}$', c='teal', linestyle='--')
    plt.axvline(x=30/Tspan, label=fr'$\frac{{{30}}}{{\mathrm{{Tspan}}}}$', c='fuchsia', linestyle='--')
    plt.xlabel('Frequencies, Hz')
    plt.ylabel('Characteristic Strain, $h_c$')
    plt.grid(which='both')
    plt.legend()
    plt.savefig(data_path+'/sc_h_c_total.svg')
    plt.show()
    plt.close()

    
    
    


