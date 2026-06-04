
import jax
import jax.numpy as jnp



#MASS MATRIX utils

def softabs_lambda(lambdas, alpha):
    """
    Compute the SoftAbs regularized eigenvalues.
    Args:
        lambdas: Eigenvalues of the Hessian.
        alpha: SoftAbs smoothing parameter.
    
    Returns:
        Regularized eigenvalues.
    """
    return lambdas / jnp.tanh(alpha * lambdas)


def softabs_metric(H, alpha = 5e-3):
    """
    Compute the SoftAbs metric tensor given a potential energy function U.
    
    Args:
        U: Potential energy function U(q).
        q: Position variable (state in phase space).
        alpha: SoftAbs regularization parameter (controls smoothness).
    
    Returns:
        SoftAbs metric g(q).
    """

    # Eigen decomposition of the Hessian
    lambdas, V = jnp.linalg.eigh(H)  # H = V D V^T, where D is diagonal of eigenvalues

    # Apply SoftAbs function to eigenvalues
    soft_lambdas = softabs_lambda(lambdas, alpha)

    # Reconstruct metric: g(q) = V Λ_soft V^T
    G = V @ jnp.diag(soft_lambdas) @ V.T

    return G
    

def make_positive_definite(A):
    A = (A + A.T) / 2  # Ensure symmetry
    eigenvalues_, eigenvectors = jnp.linalg.eigh(A)
    
    # Replace non-positive eigenvalues with a small positive number
    eigenvalues = jnp.abs(eigenvalues_)
    
    # Reconstruct the matrix
    A_positive = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
    
    return A_positive


def kinetic_energy(p, inverse_mass_matrix):
    return 0.5*jnp.dot(p.T,jnp.dot(inverse_mass_matrix,p))


def symmetrise(A):
    return (A + A.T) / 2


def compute_mass_matrix(logdensity, q):
    """ see https://arxiv.org/pdf/1212.4693"""

    mass_matrix = -jax.hessian(logdensity)(q)  # Hessian of the log-density

    mass_matrix = softabs_metric(mass_matrix)
    inverse_mass_matrix = jnp.linalg.inv(mass_matrix)
    # logdet = jnp.linalg.slogdet(mass_matrix)[1]
    
    return  inverse_mass_matrix





import jax.numpy as np
import jax
from math import pi, floor


EARTH_SEMI_MAJOR_AXIS = 6378137.0  # for ellipsoid model of Earth, in m
EARTH_SEMI_MINOR_AXIS = 6356752.314  # in m

# Constants
JULIAN_DATE_START_OF_GPS_TIME = 2444244.5
leaps = np.array([
    46828800, 78364801, 109900802, 173059203, 252028804, 315187205, 346723206, 
    393984007, 425520008, 457056009, 504489610, 551750411, 599184012, 820108813, 
    914803214, 1025136015, 1119744016, 1167264017
], dtype=np.float64)

EPOCH_J2000_0_GPS = 630763213

def GreenwichMeanSiderealTime(gpstime) :
    """Calculates Greenwich Mean Sidereal Time given GPS time."""
    return _GreenwichMeanSiderealTime(gpstime)

def _GreenwichMeanSiderealTime(gpstime):
    jd = _GPS2JD(gpstime)
    gps_ns = gpstime - np.round(gpstime)
    t_hi = (jd - 2451545.0) / 36525.0
    t_lo = gps_ns / (36525.0 * 86400.0)
    t = t_hi + t_lo

    sidereal_time = (-6.2e-6 * t + 0.093104) * t * t + 67310.54841
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * pi / 43200.0

def GPS2JD(gpstime):
    """Converts GPS time to Julian Date."""
    return _GPS2JD(gpstime)

def _GPS2JD(gpstime):
    """Helper function to compute Julian Date from GPS time."""
    dot2gps = 29224.0
    dot2utc = 2415020.5
    
    # Determine leap seconds
    nleap = jax.lax.cond(
        gpstime < 820108814,  # Condition (must be a JAX expression)
        lambda _: 32,  # If True
        lambda _:     jax.lax.cond(
                                    np.logical_and(gpstime < 914803215,gpstime >820108814),  # Condition (must be a JAX expression)
                                    lambda _: 33,  # If True
                                    lambda _: 34,   # If False
                                    operand=None),
   # If False
        operand=None
    )
#    nleap = jax.lax.cond(
#        820108814 <= gpstime < 914803215,  # Condition (must be a JAX expression)
#        lambda _: 33,  # If True
#        lambda _: 34,   # If False
#        operand=None
#    )

#    if gpstime < 820108814:
#        nleap = 32
#    elif 820108814 <= gpstime < 914803215:
#        nleap = 33
#    else:
#        nleap = 34

    dot = dot2gps + (gpstime - (nleap - 19)) / 86400.0
    utc = dot + dot2utc
    jd = utc

    return jd



def TimeDelayFromEarthCenter( lat, lon, h,  ra,dec,GPS_time,):

    def vertex(lat, lon, h):
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2 * (
            major**2 * np.cos(lat) ** 2 + minor**2 * np.sin(lat) ** 2
        ) ** (-0.5)
        x = (r + h) * np.cos(lat) * np.cos(lon)
        y = (r + h) * np.cos(lat) * np.sin(lon)
        z = ((minor / major) ** 2 * r + h) * np.sin(lat)
        return np.array([x, y, z])
  
   
    lat = np.radians(lat)
    lon = np.radians(lon)
    delta_d = - vertex(lat, lon, h)
    
    
   
    c  = 2.99792458*1e8
    gmst = GreenwichMeanSiderealTime(GPS_time) 
    
    gmst = np.mod(gmst, 2 * np.pi)
    phi = ra - gmst
    theta = np.pi / 2 - dec
    omega = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    return np.dot(omega, delta_d)/c
    





@jax.jit
def McQ2Masses(mc, q):
    """
    | Converts from chirp mass and mass ratio :math:`\\mathcal{M}_c, q` to component masses :math:`m_1, m_2`,
    | with :math:`m_1 \geq m_2` 
    
    :param mc: chirp mass in units of solar masses
    :type mc: float
    :param q: mass ratio
    :type q: float
    
    :return: :math:`m_1, m_2` in units of solar masses
    :rtype: tuple
    """
    
    factor = mc * np.power(1. + q, 1.0/5.0);
    m1     = factor * np.power(q, -3.0/5.0);
    m2     = factor * np.power(q, +2.0/5.0);
    return m1, m2

@jax.jit
def Masses2McQ(m1, m2):
    """
    | Converts from omponent masses :math:`m_1, m_2` (with :math:`m_1 \geq m_2` ) to chirp mass and mass ratio :math:`\\mathcal{M}_c, q` 
    
    :param m1: primary mass in units of solar masses
    :type m1: float
    :param m2: secondary mass in units of solar masses
    :type m2: float
    
    :return: :math:`\\mathcal{M}_c` (in units of solar masses), :math:`q`
    :rtype: tuple
    """
    
    q   = m2/m1
    eta = m1*m2/(m1+m2)
    mc  = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    return mc, q





