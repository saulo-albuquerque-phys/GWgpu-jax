import jax.numpy as jnp
import jax.random as random
import numpy as np
from functools import partial
import jax

from sharpy.utils import TimeDelayFromEarthCenter, Masses2McQ, McQ2Masses,GreenwichMeanSiderealTime
from sharpy.noise import load_data, generate_data


jax.config.update("jax_enable_x64", True) 

from ripplegw.waveforms import IMRPhenomD
from ripplegw import ms_to_Mc_eta

import sys


from astropy import constants as const
M_sun = const.M_sun.value
G = const.G.value
c = const.c.value
pc = const.pc.value




class GWDetector:
    """
    Class for a gravitational wave detector

    Arguments
    ---------
    name : string
        Name of the detector. Use GWDetector.get_detector_name to see a list of available values.
    """

    # in order, we have:
    

    def __init__(self,
                 name,
                 datafile           = None,
                 psd_file           = 'None',
                 psd_method         = 'welch',
                 T                  = 2.0,
                 trigtime           = 1126259462.4,
                 sampling_rate      = 1024,
                 flow               = 20,
                 fhigh              = 512,
                 zero_noise         = True,
                 calibration        = None,
                 download_data      = 0,
                 datalen_download   = 32,
                 channel            = '',
                 gwpy_tag           = None):
        

                   
        self.available_detectors  = {
                                        'V1': [43.631414, 10.5044968, 115.5673, 90., 51.884],
                                        'H1': [46.45514,-119.40765 , 170.9969, 90.,142.554],
                                        'L1': [30.56289, -90.774240, 242.7165, 90., -6.574],
                                        'GEO600': [52.25, -9.81, 68.775, 94.33],
                                        'TAMA300': [35.68, -139.54, 225., 90.],
                                        'ET': [40.44, 9.4566, 116.5, 60.], # Sardinia site hypothesis
                                        'K': [36.41, 137.30, 15.36, 90.]
                                    }


                 
        # initialise the needed attributes
        self.name             = name
        self.latitude         = self.available_detectors[name][0]
        self.longitude        = self.available_detectors[name][1]
    
        if name not in self.available_detectors.keys():
            raise ValueError("Not valid argument ({}) for 'name' parameter.".format(name))

        self.datafile         = datafile
        self.psd_file         = psd_file
        self.psd_method       = psd_method
        self.sampling_rate    = sampling_rate
        self.flow             = flow
        self.trigtime         = trigtime
        self.zero_noise       = zero_noise
        self.calibration      = calibration
        self.T                = T
        self.Epoch            = jnp.float64(self.trigtime -(self.T-1))
        self.download_data    = download_data
        self.datalen_download = datalen_download
        self.channel          = channel
        self.gwpy_tag         = gwpy_tag

        # set the maximum frequency cutoff to prevent aliasing
        if fhigh is None:
            self.fhigh = self.sampling_rate*0.45
            sys.stdout.write('\nMaximum frequency not given: it will be set to sampling_rate*0.45 to prevent aliasing\n')
        elif fhigh>self.sampling_rate/2.:
            self.fhigh = self.sampling_rate*0.45
            sys.stdout.write('\nMaximum frequency above the Nyquist bound: it will be set to sampling_rate*0.45 to prevent aliasing\n')
        else:
            self.fhigh = fhigh

        if self.channel is not None:
            self.Times, self.TimeSeries, self.Frequency, self.FrequencySeries, self.PowerSpectralDensity, self.mesa_object = load_data(self.datafile,
                                             self.name,
                                             chunk_size       = self.T,
                                             trigtime         = self.trigtime,
                                             sampling_rate    = self.sampling_rate,
                                             psd_file         = self.psd_file,
                                             psd_method       = self.psd_method,
                                             download_data    = self.download_data,
                                             datalen_download = self.datalen_download,
                                             channel          = self.channel,
                                             gwpy_tag         = self.gwpy_tag)
        else:
            self.Times, self.TimeSeries, self.Frequency, self.FrequencySeries, self.PowerSpectralDensity, self.mesa_object = generate_data(self.psd_file,
                                                            T = self.T,
                                                            starttime     = self.Epoch,
                                                            sampling_rate = self.sampling_rate,
                                                            fmin          = self.flow,
                                                            fmax          = self.fhigh,
                                                            zero_noise    = self.zero_noise,
                                                            asd           = False)
            
        

        # set frequency-related specifics
        self.df             = 1./self.T
        self.dt             = 1./self.sampling_rate
        self.segment_length = int(self.T*self.sampling_rate)
        self.kmin           = int(self.flow/self.df)
        self.kmax           = int(self.fhigh/self.df)+1
        
        # crop the frequency series and the frequency array
        self.FrequencySeries      = self.FrequencySeries[self.kmin:self.kmax]
        self.Frequency            = self.Frequency[self.kmin:self.kmax]
        self.PowerSpectralDensity = self.PowerSpectralDensity[self.kmin:self.kmax]
        
        # noise-weighted inner product weighting factor
        self.sigmasq              = self.PowerSpectralDensity * self.dt * self.dt
        self.TwoDeltaTOverN       = 2.0*self.dt/jnp.float64(self.segment_length)

        self.latitude   = self.available_detectors[name][0]
        self.longitude  = self.available_detectors[name][1]
        self.gamma      = self.available_detectors[name][2]
        self.zeta       = self.available_detectors[name][3]
        self.elevation  = self.available_detectors[name][4] 


from flax import struct
@struct.dataclass(frozen = False)
class Detector:
    Frequency       :      jnp.ndarray
    FrequencySeries :      jnp.ndarray
    PowerSpectralDensity: jnp.ndarray
    sigmasq: jnp.ndarray
    latitude: float
    longitude: float    
    elevation: float
    gamma: float
    zeta: float
    TwoDeltaTOverN: float
    T: float
    trigtime: float



def stack_detectors(detectors_list):
    
    return Detector(
        Frequency           = jnp.stack([d["Frequency"] for d in detectors_list]),
        FrequencySeries     = jnp.stack([d["FrequencySeries"] for d in detectors_list]),
        TwoDeltaTOverN      = jnp.stack([d["TwoDeltaTOverN"] for d in detectors_list]),
        sigmasq             = jnp.stack([d["sigmasq"] for d in detectors_list]),
        latitude            = jnp.array([d["latitude"] for d in detectors_list]),
        longitude           = jnp.array([d["longitude"] for d in detectors_list]),
        elevation           = jnp.array([d["elevation"] for d in detectors_list]),
        gamma               = jnp.array([d["gamma"] for d in detectors_list]),
        zeta                = jnp.array([d["zeta"] for d in detectors_list]),
        PowerSpectralDensity= jnp.stack([d["PowerSpectralDensity"] for d in detectors_list]),
        T                   = jnp.array([d["T"] for d in detectors_list]),
        trigtime            = jnp.array([d["trigtime"] for d in detectors_list])

    )



def inject_signal_in_detector(params,evaluated_template,  detector_dictionary):
        """
        Inject a signal into the detector noise.
        """
        h           = project_waveform(params,evaluated_template, detector_dictionary)

        # add to the detector noise
        detector_dictionary.FrequencySeries += h

        df          = detector_dictionary.Frequency[1] - detector_dictionary.Frequency[0]

        # signal-to-noise ratio
        SNR         = jnp.sqrt(4.0*df*jnp.sum(jnp.conj(h)*h/detector_dictionary.PowerSpectralDensity).real)
        
        return detector_dictionary, SNR



class GWNetwork:
    """
    Class for a network of gravitational wave detectors

    Arguments
    ---------
    detectors : list
        List of GWDetector objects.
    """
    def __init__(self, detectors_setting,
                       injection_parameters = None,
                        ):
        
        self.detectors_settings = detectors_setting
        self.injection_parameters = injection_parameters
        self.list_of_detectors_necessary_parameters = [ "FrequencySeries", 
                                                        "TwoDeltaTOverN", 
                                                        "sigmasq",
                                                        "Frequency", 
                                                        "latitude", 
                                                        "longitude",
                                                        "gamma",
                                                        "zeta", 
                                                        "elevation", 
                                                        "PowerSpectralDensity",
                                                        "T", 
                                                        "trigtime"

                                        ]
        
        self.detectors        = self.detector_constructor()
        
        self.batched_detector = self.construct_batched_detectors()

        if self.injection_parameters is not None:
            print("Injecting signal with parameters: ", self.injection_parameters)
            self.batched_detector, self.snr = self.inject_signal_in_noise()
            print("Injected signal with SNR: ", self.snr)


    def inject_signal_in_noise(self, ):
        evaluated_template = template(self.injection_parameters, self.batched_detector.Frequency[0]) # Assuming all detectors have the same frequency array
        detector_dictionaries, snr = jax.vmap(inject_signal_in_detector,in_axes=(None,None, 0))(self.injection_parameters,evaluated_template, self.batched_detector)
        total_snr = jnp.sqrt(jnp.sum(snr**2))
        return detector_dictionaries, total_snr
    



    def detector_constructor(self):
       
        detector_names = self.detectors_settings.keys()


               
        self.detectors = [GWDetector(name, 
                                    channel          = self.detectors_settings[name]['channel'] if 'channel' in self.detectors_settings[name] else None, 
                                    psd_file         = self.detectors_settings[name]['psd_file'] if 'psd_file' in self.detectors_settings[name] else None,
                                    datafile         = self.detectors_settings[name]['data_file'] if 'data_file' in self.detectors_settings[name] else None,
                                    T                = self.detectors_settings[name]['duration'] if 'duration' in self.detectors_settings[name] else 2.0,
                                    sampling_rate    = self.detectors_settings[name]['sampling_rate'] if 'sampling_rate' in self.detectors_settings[name] else 1024,
                                    flow             = self.detectors_settings[name]['f_lower'] if 'f_lowr' in self.detectors_settings[name] else 20.0,
                                    fhigh            = self.detectors_settings[name]['f_upper'] if 'f_high' in self.detectors_settings[name] else 512.0,
                                    trigtime         = self.detectors_settings[name]['trigger_time'] if 'trigger_time' in self.detectors_settings[name] else 1126259462.4,
                                    download_data    = self.detectors_settings[name]['download_data'] if 'download_data' in self.detectors_settings[name] else False, 
                                    datalen_download = self.detectors_settings[name]['datalen_download'] if 'datalen_download' in self.detectors_settings[name] else 32, 
                                    zero_noise       = self.detectors_settings[name]['zero_noise'] if 'zero_noise' in self.detectors_settings[name] else False,

                                          ).__dict__ for name in detector_names
                        ]


                  
        return self.detectors
    
    def construct_batched_detectors(self):
        "construct a batched detector for jax.vmap"

        detectors_list  = [{} for i in self.detectors]
        for i, dict in enumerate(detectors_list):
            for key in self.list_of_detectors_necessary_parameters:
                dict[key] = self.detectors[i][key]
                
        
        self.batched_detector = stack_detectors(detectors_list)

        return self.batched_detector
    





def project_waveform(params,evaluated_template, detector_dictionary):
    
    f = detector_dictionary.Frequency

    # h_plus, h_cross = template(params, f)
    h_plus, h_cross   = evaluated_template
    # h_plus, h_cross   = TaylorF2(params, f)




    latitude        = detector_dictionary.latitude
    longitude       = detector_dictionary.longitude
    gamma           = detector_dictionary.gamma
    zeta            = detector_dictionary.zeta
    elevation       = detector_dictionary.elevation
    trigger_time    = detector_dictionary.trigtime

    
    fplus, fcross   = antenna_pattern_functions(params, latitude, longitude, gamma, zeta, trigger_time)


    ra = params[0]
    dec = params[1]
    tc  = trigger_time + params[8]

    timedelay       = TimeDelayFromEarthCenter(latitude, longitude, elevation, ra, dec, tc)
    
    timeshift       = timedelay
    timeshift       = timeshift + (params[8] + (detector_dictionary.T - 1) )
    
    shift           = 2.0*np.pi*f*timeshift

  
    h = (fplus*h_plus + fcross*h_cross)*(jnp.cos(shift)-1j*jnp.sin(shift))
    return h


def antenna_pattern_functions(params, det_latitute, det_longitude, det_gamma, det_zeta, trigger_time):
    '''
    #    default_names = ['phiref','ra','dec','tc','mc','q','costheta_jn','psi','logdistance']
    Evaluate the antenna pattern functions.

    :param right_ascension: float
        Right ascension of the source in degree.

    :param declination: float
        Declination of the source in degree.

    :param polarization: float
        Polarization angle of the wave in degree.

    :param GPS_time: float, int, list or np.ndarray
        time of arrival of the source signal.

    :return: tuple of float or np.ndarray
        fplus and fcross.
    '''

    ra = params[0]
    dec = params[1]

    pol = params[5]
    
    tc  = trigger_time + params[8]
    lat = jnp.radians(det_latitute)
    g_ = jnp.radians(det_gamma)
    z_ = jnp.radians(det_zeta)
    gmst = jnp.mod(GreenwichMeanSiderealTime(tc), 2*jnp.pi)
    lst = gmst + jnp.radians(det_longitude)
    ampl11, ampl12 = _ab_factors(g_, lat, ra, dec, lst)

    c2pol = jnp.cos(2*pol)
    s2pol = jnp.sin(2*pol)
    
    fplus = jnp.sin(z_)*(ampl11*c2pol + ampl12*s2pol)
    fcross = jnp.sin(z_)*(ampl12*c2pol - ampl11*s2pol)

    return fplus, fcross

# @jax.jit
def _ab_factors(g_, lat, ra, dec, lst):
    """
    Method that calculates the amplitude factors of plus and cross
    polarization in the wave projection on the detector.
    :param g_: float
        this represent the orientation of the detector's arms with respect to local geographical direction, in
        rad. It is measured counterclock-wise from East to the bisector of the interferometer arms.
    :param lat: float
        longitude of the detector in rad.
    :param ra: float
        Right ascension of the source in rad.
    :param dec: float
        Declination of the source in rad.
    :param lst: float or ndarray
        Local sidereal time(s) in rad.
    :return: tuple of float or np.ndarray
        relative amplitudes of hplus and hcross.
    """
    s2g = jnp.sin(2*g_)
    c2g = jnp.cos(2*g_)
    cdec  = jnp.cos(dec)
    sdec  = jnp.sin(dec)
    c2dec = jnp.cos(2*dec)
    s2dec = jnp.sin(2*dec)
    clat  = jnp.cos(lat)
    slat  = jnp.sin(lat)
    c2lat = jnp.cos(2*lat)
    s2lat = jnp.sin(2*lat)
    ra_lst = ra - lst
    cra_lst = jnp.cos(ra_lst)
    sra_lst = jnp.sin(ra_lst)
    two_ra_lst = 2*ra_lst
    c2ra_lst = jnp.cos(two_ra_lst)
    s2ra_lst = jnp.sin(two_ra_lst)
    
    a_ = (1/16)*s2g*(3-c2lat)*(3-c2dec)*c2ra_lst-\
         (1/4)*c2g*slat*(3-c2dec)*s2ra_lst+\
         (1/4)*s2g*s2lat*s2dec*cra_lst-\
         (1/2)*c2g*clat*s2dec*sra_lst+\
         (3/4)*s2g*(clat**2)*(cdec**2)

    b_ = c2g*slat*sdec*c2ra_lst+\
         (1/4)*s2g*(3-c2lat)*sdec*s2ra_lst+\
                 c2g*clat*cdec*cra_lst+\
         (1/2)*s2g*s2lat*cdec*sra_lst


    return a_, b_


###DO NOT USE THIS FUNCTION ANYMORE
###Only for testing purposes
def TaylorF2(params, frequency_array):

    Mc, q, phi_c, logdistance, cos_iota= params[6], params[7], params[4], params[2], params[3]

    distance = jnp.exp(logdistance)
    
    nu = q / ((1 + q) ** 2)

    Mc *= M_sun
    r = distance * pc * 1e6  # Convert to Megaparsec

    M = Mc / (nu ** (3 / 5))
    f_lso = frequency_array[-1] / 2

    # Precompute terms
    pi_M = G * jnp.pi * M
    v = jnp.power(pi_M * frequency_array, 1/3) / c
    v_lso = jnp.power(pi_M * f_lso, 1/3) / c
    gamma = jnp.euler_gamma

    # Compute amplitude
    amp = jnp.power(jnp.pi, -2/3) * jnp.sqrt(5/24) * jnp.power(G * Mc / c**3, 5/6) \
          * jnp.power(frequency_array, -7/6) * (c / r)

    # Compute phase terms (factorized and precomputed where possible)
    v2 = v**2
    v3 = v**3
    v4 = v**4
    v5 = v**5
    v6 = v**6
    v7 = v**7
    log_v = jnp.log(v)

    phi_plus = (3 / (128 * nu * v**5)) * (1 +
        v2 * (20/9) * (743/336 + nu * 11/4) -
        v3 * (16 * jnp.pi) +
        v4 * (10 * (3058673/1016064 + nu * 5429/1008 + (nu**2) * 617/144)) +
        v5 * jnp.pi * (38645/756 - nu * 65/9) * (1 + 3 * log_v) +
        v6 * (11583231236531/4694215680 - jnp.pi**2 * 640/3 - 6848 * gamma/21 - 6848/21 * log_v +
              nu * (-15737765635/3048192 + 2255 * (jnp.pi**2) / 12) + nu**2 * 76055/1728 - nu**3 * 127825/1296) +
        v7 * jnp.pi * (77096675/254016 + nu * 378515/1512 - nu**2 * 74045/756)
    )

    phi_plus += jnp.pi - jnp.pi / 4
    phi_cross = phi_plus + jnp.pi / 2

    # Compute phase factor
    phase_factor = jnp.exp(-1j * phi_c)
    exp_phi_plus = jnp.exp(1j * phi_plus)
    exp_phi_cross = jnp.exp(1j * phi_cross)

    # Compute strain polarizations
    #  cos_iota = jnp.cos(iota)
    cos_iota_sq = cos_iota**2

    h_plus = phase_factor * amp * ((1 + cos_iota_sq) / 2) * exp_phi_plus
    h_cross = phase_factor * amp * cos_iota * exp_phi_cross

    return h_plus, h_cross




# @jax.jit
def template(params, frequency_array):
    mc                      = params[6]
    q                       = params[7]
    m1_msun, m2_msun        = McQ2Masses(mc, q)
    chi1                    = params[9] # Dimensionless spin
    chi2                    = params[10]
    tc                      = 0.0 # Time of coalescence in seconds
    phic                    = params[4] # Phase of coalescence
    dist_mpc                = jnp.exp(params[2]) # Distance to source in Mpc
    inclination             = params[3] # Inclination Angle

    # The PhenomD waveform model is parameterized with the chirp mass and symmetric mass ratio
    Mc, eta           = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

    theta_ripple      = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])
    # hp, hc       = IMRPhenomD.gen_IMRPhenomD_hphc(frequency_array, theta_ripple, frequency_array[0]) 
    hp, hc            = jax.vmap(IMRPhenomD.gen_IMRPhenomD_hphc, in_axes=(0, None, None))(jnp.array([frequency_array]), theta_ripple, 20)

    # jax.debug.print("Max hp: {}, Max hc: {}", jnp.max(jnp.abs(hp)), jnp.max(jnp.abs(hc)))
    return hp, hc 







######Likelihood 







def log_likelihood_det(params, detector_list):
    evaluated_template = template(params, detector_list.Frequency[0]) # Assuming all detectors have the same frequency array

    log_likelihoods = jax.vmap(single_detector_log_likelihood, in_axes=(None,None, 0))(params,evaluated_template,  detector_list)

    # Then use jnp.sum
    return jnp.sum(log_likelihoods)


def single_detector_log_likelihood(params, evaluated_template, detector_dictionary):

    h = project_waveform(params, evaluated_template, detector_dictionary)
    residuals = detector_dictionary.FrequencySeries - h
    return -detector_dictionary.TwoDeltaTOverN * jnp.vdot(residuals / jnp.sqrt(detector_dictionary.sigmasq), residuals / jnp.sqrt(detector_dictionary.sigmasq)).real