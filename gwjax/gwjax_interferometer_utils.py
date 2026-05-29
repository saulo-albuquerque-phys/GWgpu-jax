"""
gwjax_interferometer_utils.py
==============================
Ground-based gravitational-wave interferometer class.

Each :class:`Interferometer` instance bundles:

  * **Geometry** — geocentric vertex position and arm unit vectors (ECEF).
  * **PSD** — power spectral density on the shared frequency grid.
  * **Strain data** — time- and frequency-domain data arrays, either loaded
    from an external source or generated as a random noise realisation.

Supported detector names (geometry pre-loaded)
------------------------------------------------
  H1  — LIGO Hanford        L1  — LIGO Livingston
  V1  — Virgo               K1  — KAGRA
  ET  — Einstein Telescope  CE  — Cosmic Explorer (approximate location)

Custom detectors can be constructed by passing ``lat``, ``lon``, ``elev``,
``xarm_az``, ``yarm_az``, and ``arm_length`` as keyword arguments.

Noise generation
----------------
The one-sided PSD S_n(f) [Hz^{-1}] is used to draw a coloured Gaussian noise
realisation via:

    h̃_k  =  sqrt( S_n(f_k) / (4 · df) )  ×  (n_re + i·n_im),  n ~ N(0,1)

    h(t)  =  IRFFT( h̃_k × fs )

which satisfies  E[|h̃_k|²] = S_n(f_k)/(2·df)  and
                 Var(h(t))  = ∫₀^{fs/2} S_n(f) df.

Antenna pattern
---------------
:meth:`antenna_pattern` returns (F₊, F×) using the standard detector-tensor
formalism.  Arm vectors are rotated from ECEF to the geocentric celestial
frame via the Greenwich Mean Sidereal Time (GMST) before computing the dot
products with the sky-polarisation basis vectors.

Time delay
----------
:meth:`time_delay_from_geocenter` returns Δt [s] such that
  t_detector = t_geocenter + Δt.
A positive Δt means the signal arrives *after* the geocenter (detector is
further from the source).

Waveform projection & injection
-------------------------------
:meth:`waveform_projection` projects (h+, h×) onto the detector.
:meth:`inject_signal` adds a projected waveform to the data.

Likelihood
----------
:meth:`log_likelihood` computes the Gaussian log-likelihood -0.5 ⟨d-h|d-h⟩.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
import jax
import jax.numpy as jnp

from gwjax.gwjax_timefrequencydomain_utils import TimeFrequencyGrid
from gwjax.gwjax_psd_utils import get_psd, _PSD_REGISTRY
from gwjax.gwjax_likelihood_utils import (
    waveform_projection_fd,
    log_likelihood_ifo,
    optimal_snr_squared,
    precompute_weight,
)

# ── Physical constants ────────────────────────────────────────────────────────
_C_LIGHT = 299_792_458.0          # speed of light [m/s]
_WGS84_A = 6_378_137.0            # semi-major axis [m]
_WGS84_E2 = 6.694_379_990_14e-3   # first eccentricity squared

# ── Known detector parameters ────────────────────────────────────────────────
#
# Geometry source of truth: LAL/bilby reference values (`bilby_cython`'s
# ``InterferometerGeometry``). The arm ECEF unit vectors are stored *directly*
# rather than reconstructed from (lat, lon, azimuth, tilt) so the geometry
# matches bilby bit-for-bit (no projection rounding, no missing arm-tilt
# correction). Reconstruction from (lat, lon, az) is kept for *custom* user
# detectors — see :func:`_azimuth_to_ecef`.
#
# Keys
#   lat, lon     : degrees (positive N / positive E) — informational only
#   elev         : metres above WGS84 ellipsoid     — informational only
#   vertex_ecef  : ECEF position of the vertex [m]
#   xarm_ecef    : ECEF unit vector along the x-arm (LAL-precision)
#   yarm_ecef    : ECEF unit vector along the y-arm (LAL-precision)
#   arm_len      : arm length [m]
#   psd          : default PSD model name
#
# The N-CW (xarm_az, yarm_az) values are *retained* for reference and for
# the custom-detector path (``_azimuth_to_ecef``); they are ignored when
# ``xarm_ecef``/``yarm_ecef`` are present.
_KNOWN_DETECTORS: dict[str, dict] = {
    "H1": dict(
        lat=46.4547471372, lon=-119.4076509511, elev=142.554,
        xarm_az=324.0006, yarm_az=234.0006, arm_len=3995.1, psd="aLIGO",
        vertex_ecef=(-2161414.926360, -3834695.178888, 4600350.226639),
        xarm_ecef  =(-0.2238926615,    0.7998306275,    0.5569048783),
        yarm_ecef  =(-0.9139781857,    0.0260940399,   -0.4049234212),
    ),
    "L1": dict(
        lat=30.5628943333, lon=-90.7742403889,  elev=-6.574,
        xarm_az=252.2835, yarm_az=342.2835, arm_len=3994.5, psd="aLIGO",
        vertex_ecef=(   -74276.044724, -5496283.719708, 3224257.017436),
        xarm_ecef  =(-0.9545741215,   -0.1415807734,   -0.2621891132),
        yarm_ecef  =( 0.2977415689,   -0.4879103365,   -0.8205446129),
    ),
    "V1": dict(
        lat=43.6314144722, lon=10.5044966111,   elev=51.884,
        xarm_az=19.4326,  yarm_az=289.4326, arm_len=3000.0, psd="AdV",
        vertex_ecef=(4546374.099003,   842989.697626,  4378576.962409),
        xarm_ecef  =(-0.7004582148,    0.2084894862,    0.6825616628),
        yarm_ecef  =(-0.0537925537,   -0.9690818055,    0.2408045171),
    ),
    "K1": dict(
        lat=36.4118603389, lon=137.3059560306, elev=414.181,
        xarm_az=29.6038,  yarm_az=119.6036, arm_len=3000.0, psd="KAGRA",
        vertex_ecef=(-3777336.023929, 3484898.410986, 3765313.696691),
        xarm_ecef  =(-0.3759040795,  -0.8361583410,   0.3994187675),
        yarm_ecef  =( 0.7164379022,   0.0111408152,   0.6975619073),
    ),
    "GEO600": dict(
        lat=52.2451466667, lon=9.8071927778,   elev=114.425,
        xarm_az=21.6117,  yarm_az=115.9431, arm_len=600.0,  psd="aLIGO",
        vertex_ecef=(3856309.949259,  666598.956317,  5019641.417249),
        xarm_ecef  =(-0.4453067690,   0.8665135413,   0.2255131131),
        yarm_ecef  =(-0.6260575678,  -0.5521860952,   0.5505837249),
    ),
    # Einstein Telescope: 10 km arms, 60° opening angle, Sardinia placeholder
    # (not in bilby's registry — kept as geometric placeholder via lat/lon/az)
    "ET": dict(lat=40.522,     lon=9.425,       elev=0.0,
               xarm_az=70.5674, yarm_az=130.5674, arm_len=10000.0, psd="ET_D"),
    # Cosmic Explorer: 20 km arms, LHO-placeholder geometry (LAL azimuths)
    "CE": dict(lat=46.4547471372, lon=-119.4076509511, elev=142.554,
               xarm_az=324.0006, yarm_az=234.0006, arm_len=20000.0, psd="CE",
               vertex_ecef=(-2161414.926360, -3834695.178888, 4600350.226639),
               xarm_ecef  =(-0.2238926615,    0.7998306275,    0.5569048783),
               yarm_ecef  =(-0.9139781857,    0.0260940399,   -0.4049234212)),
}


# ── Geodetic helper functions ─────────────────────────────────────────────────

def _geodetic_to_ecef(lat_deg: float, lon_deg: float, elev_m: float) -> np.ndarray:
    """Convert geodetic coordinates to ECEF Cartesian [m]."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    x = (N + elev_m) * math.cos(lat) * math.cos(lon)
    y = (N + elev_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + elev_m) * math.sin(lat)
    return np.array([x, y, z])


def _azimuth_to_ecef(lat_deg: float, lon_deg: float, az_deg: float) -> np.ndarray:
    """Convert a horizontal unit vector (given by azimuth from North) to ECEF.

    The ENU (East, North, Up) representation of the arm unit vector is::

        arm_ENU = [sin(az), cos(az), 0]

    which is then rotated to geocentric ECEF.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    az  = math.radians(az_deg)
    # ENU components
    e_e = math.sin(az)
    e_n = math.cos(az)
    # Rotation matrix: ENU → ECEF
    # [East ]   [-sin(lon),           cos(lon),          0       ]
    # [North] = [-sin(lat)*cos(lon),  -sin(lat)*sin(lon), cos(lat)]
    # [Up   ]   [ cos(lat)*cos(lon),   cos(lat)*sin(lon), sin(lat)]
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    x = -slon * e_e + (-slat * clon) * e_n
    y =  clon * e_e + (-slat * slon) * e_n
    z =               clat           * e_n
    return np.array([x, y, z])


# ── Main class ────────────────────────────────────────────────────────────────

class Interferometer:
    """Single ground-based gravitational-wave interferometer.

    Parameters
    ----------
    name : str
        Detector name.  If it matches a key in ``_KNOWN_DETECTORS`` the
        geometry is loaded automatically.  Unknown names require the geometry
        keyword arguments below.
    grid : TimeFrequencyGrid, optional
        Shared time/frequency grid.  Must be provided before calling
        :meth:`set_psd`, :meth:`generate_noise`, or :meth:`load_data`.
    psd_model : str | array | None
        PSD model.  Accepts a model name (``"aLIGO"``, ``"AdV"``, etc.), a
        path to a two-column ASCII file, or a JAX/NumPy array of length
        ``grid.n_samples // 2 + 1``.  If *None* the default model for the
        detector is used.  The PSD is evaluated (or loaded) immediately when
        *grid* is provided.
    lat, lon : float, optional
        Geodetic latitude / longitude [degrees].  Required for custom detectors.
    elev : float, optional
        Elevation above WGS84 ellipsoid [m].
    xarm_az, yarm_az : float, optional
        Azimuth of x-arm / y-arm from North, measured clockwise [degrees].
    arm_len : float, optional
        Arm length [m].

    Attributes
    ----------
    name : str
    vertex : jnp.ndarray, shape (3,)
        Geocentric ECEF position of the vertex [m].
    xarm, yarm : jnp.ndarray, shape (3,)
        ECEF unit vectors along each arm.
    arm_length : float  [m]
    grid : TimeFrequencyGrid or None
    psd : jnp.ndarray or None
        One-sided PSD on ``grid.frequency_domain_array`` [Hz^{-1}].
    strain_data_fd : jnp.ndarray or None
        Complex FD strain on ``grid.frequency_domain_array`` [Hz^{-1/2}].
        Convention: h̃(f_k) = rfft(h)[k] × dt.
    strain_data_td : jnp.ndarray or None
        Real TD strain on ``grid.time_domain_array`` [dimensionless].
    """

    def __init__(
        self,
        name: str,
        grid: TimeFrequencyGrid | None = None,
        psd_model: str | jnp.ndarray | None = None,
        *,
        lat: float | None = None,
        lon: float | None = None,
        elev: float = 0.0,
        xarm_az: float | None = None,
        yarm_az: float | None = None,
        arm_len: float = 4000.0,
    ) -> None:
        self.name = name

        # ── Geometry ──────────────────────────────────────────────────────────
        if name in _KNOWN_DETECTORS:
            cfg = _KNOWN_DETECTORS[name]
            _lat, _lon, _elev = cfg["lat"], cfg["lon"], cfg["elev"]
            _xaz, _yaz = cfg["xarm_az"], cfg["yarm_az"]
            _arm_len = cfg["arm_len"]
            _default_psd = cfg["psd"]
        else:
            if lat is None or lon is None or xarm_az is None or yarm_az is None:
                raise ValueError(
                    f"Detector '{name}' is not in the known-detector list. "
                    "Provide lat, lon, xarm_az, yarm_az (and optionally "
                    "elev, arm_len)."
                )
            _lat, _lon, _elev = lat, lon, elev
            _xaz, _yaz = xarm_az, yarm_az
            _arm_len = arm_len
            _default_psd = "aLIGO"

        # Geometry source preference:
        #   1. If known-detector entry carries explicit ECEF vectors
        #      (vertex_ecef + xarm_ecef + yarm_ecef), use them verbatim —
        #      this matches LAL/bilby to all published digits and accounts
        #      for arm tilts that the lat/lon/azimuth-only path ignores.
        #   2. Otherwise reconstruct from geodetic lat/lon + N-CW azimuth.
        cfg = _KNOWN_DETECTORS.get(name, {})
        if {"vertex_ecef", "xarm_ecef", "yarm_ecef"} <= cfg.keys():
            self.vertex = jnp.array(cfg["vertex_ecef"], dtype=jnp.float64)
            self.xarm   = jnp.array(cfg["xarm_ecef"],   dtype=jnp.float64)
            self.yarm   = jnp.array(cfg["yarm_ecef"],   dtype=jnp.float64)
        else:
            self.vertex = jnp.array(_geodetic_to_ecef(_lat, _lon, _elev))
            self.xarm   = jnp.array(_azimuth_to_ecef(_lat, _lon, _xaz))
            self.yarm   = jnp.array(_azimuth_to_ecef(_lat, _lon, _yaz))
        self.arm_length = float(_arm_len)
        self._lat       = _lat
        self._lon       = _lon

        # ── Grid and PSD ──────────────────────────────────────────────────────
        self.grid: TimeFrequencyGrid | None = None
        self.psd:  jnp.ndarray | None = None
        self._psd_model = psd_model if psd_model is not None else _default_psd
        self._weight: jnp.ndarray | None = None  # cached inner-product weight
        self.strain_data_fd: jnp.ndarray | None = None
        self.strain_data_td: jnp.ndarray | None = None

        if grid is not None:
            self.set_grid(grid, psd_model=self._psd_model)

    # ── Grid / PSD management ─────────────────────────────────────────────────

    def set_grid(
        self,
        grid: TimeFrequencyGrid,
        psd_model: str | jnp.ndarray | None = None,
    ) -> None:
        """Attach a :class:`TimeFrequencyGrid` and (re-)evaluate the PSD.

        If ``psd_model`` is ``None`` the previously cached model (set at
        construction or by the last call to :meth:`set_psd`) is re-evaluated
        on the new grid, so ``self.psd`` is always consistent with
        ``grid.frequency_domain_array``.

        Parameters
        ----------
        grid : TimeFrequencyGrid
        psd_model : str | array | None
            Override the cached model. Passing an array requires its shape
            to match the new grid.
        """
        self.grid = grid
        self._weight = None  # mask/df may have changed
        model = psd_model if psd_model is not None else self._psd_model
        if model is not None:
            self.set_psd(model)

    def set_psd(self, model: str | jnp.ndarray, **kwargs) -> None:
        """Evaluate the PSD on the current frequency grid.

        Parameters
        ----------
        model : str | array
            Model name, filepath, or custom array.  See :func:`get_psd`.
        **kwargs
            Extra keyword arguments forwarded to the analytical function
            (e.g. ``f_min=20.0``).
        """
        if self.grid is None:
            raise RuntimeError("Attach a TimeFrequencyGrid first via set_grid().")
        self.psd = get_psd(model, self.grid.frequency_domain_array, **kwargs)
        self._psd_model = model
        self._weight = None  # invalidate cached weight

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(
        self,
        data:   jnp.ndarray,
        domain: str = "td",
    ) -> None:
        """Load strain data from an external array.

        Parameters
        ----------
        data : array-like
            * If *domain* is ``"td"``: real array of shape ``(N,)``.
            * If *domain* is ``"fd"``: complex array of shape ``(N//2+1,)``.
        domain : ``"td"`` or ``"fd"``
            Domain of the supplied data.

        Notes
        -----
        Both ``strain_data_td`` and ``strain_data_fd`` are populated using
        the FFT pair convention h̃(f_k) = rfft(h)[k] × dt.
        """
        if self.grid is None:
            raise RuntimeError("Attach a TimeFrequencyGrid first via set_grid().")
        dt = self.grid.dt
        fs = self.grid.sampling_rate
        N  = self.grid.n_samples

        if domain == "td":
            self.strain_data_td = jnp.asarray(data)
            self.strain_data_fd = jnp.fft.rfft(self.strain_data_td) * dt
        elif domain == "fd":
            self.strain_data_fd = jnp.asarray(data)
            self.strain_data_td = jnp.fft.irfft(
                self.strain_data_fd * fs, n=N
            ).real
        else:
            raise ValueError(f"domain must be 'td' or 'fd', got '{domain}'.")

    # ── Noise generation ──────────────────────────────────────────────────────

    def generate_noise(self, seed: int | jnp.ndarray = 42) -> None:
        """Draw a coloured Gaussian noise realisation from the design PSD.

        Populates ``strain_data_fd`` and ``strain_data_td``.

        Parameters
        ----------
        seed : int or JAX PRNGKey
            Random seed.  An integer is converted to a JAX PRNGKey via
            ``jax.random.PRNGKey(seed)``.

        Notes
        -----
        Normalisation::

            h̃_k  =  sqrt( S_n(f_k) / (4 · df) )  ×  (n₁_k + i·n₂_k)

        where n₁, n₂ ~ N(0, 1) independently.  This gives

            E[|h̃_k|²]  =  S_n(f_k) / (2 · df)

        consistent with the one-sided PSD convention
        ⟨h̃(f) h̃*(f')⟩ = S_n(f)/2 · δ(f-f').
        """
        if self.grid is None:
            raise RuntimeError("Attach a TimeFrequencyGrid first via set_grid().")
        if self.psd is None:
            raise RuntimeError("Set a PSD first via set_psd().")

        key = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
        key1, key2 = jax.random.split(key)

        nf = len(self.grid.frequency_domain_array)
        # Replace inf/NaN PSD bins (e.g. below f_min) with zero amplitude so
        # the strain stays finite everywhere. Out-of-band bins are still
        # ignored by the analysis mask in the inner product.
        safe_psd = jnp.where(jnp.isfinite(self.psd), self.psd, 0.0)
        amp = jnp.sqrt(safe_psd / (4.0 * self.grid.df))

        self.strain_data_fd = amp * (
            jax.random.normal(key1, (nf,))
            + 1j * jax.random.normal(key2, (nf,))
        )
        self.strain_data_td = jnp.fft.irfft(
            self.strain_data_fd * self.grid.sampling_rate,
            n=self.grid.n_samples,
        ).real

    # ── Antenna pattern ───────────────────────────────────────────────────────

    def antenna_pattern(
        self,
        ra:   float | jnp.ndarray,
        dec:  float | jnp.ndarray,
        psi:  float | jnp.ndarray,
        gmst: float | jnp.ndarray = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the antenna response patterns F₊ and F×.

        Parameters
        ----------
        ra   : float [rad]  Right ascension of the source.
        dec  : float [rad]  Declination of the source.
        psi  : float [rad]  Gravitational-wave polarisation angle.
        gmst : float [rad]  Greenwich Mean Sidereal Time. Pass the trigger
                            GMST (as written to ``network.gmst`` by
                            :func:`gwjax.compat.attach_event_to_network`)
                            to interpret ``ra`` in the celestial J2000 frame.
                            Defaults to 0 (Earth-rotating ECEF frame —
                            appropriate for synthetic injections where
                            inject and recover cancel by symmetry).

        Returns
        -------
        Fp, Fc : scalar JAX arrays
            Plus and cross antenna response factors.

        Notes
        -----
        The x/y arm unit vectors stored in ECEF are rotated into the geocentric
        celestial frame by the GMST angle before computing the dot products.
        The source-frame basis vectors (m, n) follow the LAL/bilby
        right-handed wave-frame convention::

            m = ( sin(ra−gmst),  −cos(ra−gmst),  0)
            n = (−sin(dec)·cos(ra−gmst),  −sin(dec)·sin(ra−gmst),  cos(dec))

        with polarisation tensors ``e+ = m⊗m − n⊗n`` and
        ``e× = m⊗n + n⊗m``. The detector tensor is
        D = (x̂⊗x̂ − ŷ⊗ŷ)/2, so ``Fp = D : e+(ψ)`` and ``Fc = D : e×(ψ)``
        with the standard 2ψ rotation applied to the polarisation basis.
        The convention is validated against bilby's
        ``Interferometer.antenna_response`` to better than 7·10⁻⁴ across a
        grid of (ra, dec, ψ) for H1/L1/V1.
        """
        # Rotate arm vectors from ECEF to geocentric celestial frame
        cos_gmst = jnp.cos(gmst)
        sin_gmst = jnp.sin(gmst)
        # R_z(+gmst): ECEF → celestial
        xarm = jnp.array([
            self.xarm[0] * cos_gmst - self.xarm[1] * sin_gmst,
            self.xarm[0] * sin_gmst + self.xarm[1] * cos_gmst,
            self.xarm[2],
        ])
        yarm = jnp.array([
            self.yarm[0] * cos_gmst - self.yarm[1] * sin_gmst,
            self.yarm[0] * sin_gmst + self.yarm[1] * cos_gmst,
            self.yarm[2],
        ])

        # Source-sky basis vectors in celestial frame
        cos_dec = jnp.cos(dec)
        sin_dec = jnp.sin(dec)
        ra_eff  = ra        # already in celestial frame
        cos_ra  = jnp.cos(ra_eff)
        sin_ra  = jnp.sin(ra_eff)

        # LAL right-handed wave frame: m = (sin α, −cos α, 0).
        # NB. an earlier version of this code used m = (−sin α, cos α, 0)
        # which gave the correct F+ but the WRONG SIGN on F×. The error
        # is invisible to inject↔recover round-trips (symmetric) but
        # biases real-data PE because the strain is recorded in the LAL
        # convention. Validated against bilby on GW150914.
        m = jnp.array([ sin_ra,              -cos_ra,             0.0])
        n = jnp.array([-sin_dec * cos_ra,    -sin_dec * sin_ra,   cos_dec])

        # Unrotated detector response (ψ = 0)
        xm, xn = jnp.dot(xarm, m), jnp.dot(xarm, n)
        ym, yn = jnp.dot(yarm, m), jnp.dot(yarm, n)
        D_plus  = 0.5 * ((xm ** 2 - xn ** 2) - (ym ** 2 - yn ** 2))
        D_cross = xm * xn - ym * yn

        # Rotate by polarisation angle ψ (standard LAL form):
        #   e+(ψ) =  cos(2ψ) e+ + sin(2ψ) e×
        #   e×(ψ) = -sin(2ψ) e+ + cos(2ψ) e×
        c2psi = jnp.cos(2.0 * psi)
        s2psi = jnp.sin(2.0 * psi)
        Fp =  c2psi * D_plus + s2psi * D_cross
        Fc = -s2psi * D_plus + c2psi * D_cross
        return Fp, Fc

    # ── Time delay ────────────────────────────────────────────────────────────

    def time_delay_from_geocenter(
        self,
        ra:   float | jnp.ndarray,
        dec:  float | jnp.ndarray,
        gmst: float | jnp.ndarray = 0.0,
    ) -> jnp.ndarray:
        """Compute the arrival-time delay relative to the geocenter [s].

        Returns ``Δt = t_det − t_geo = −(n̂ · vertex) / c`` where n̂ points
        from the geocenter toward the source (ra, dec). A **positive** value
        means the signal arrives at the detector *after* the geocenter (the
        detector is "downstream" in the wave-propagation direction). A
        **negative** value means the detector is closer to the source and
        receives the signal first.

        Combined with the FD shift ``exp(−2πi · f · Δt)`` in
        :func:`gwjax.gwjax_likelihood_utils.waveform_projection_fd`, this
        is the SHARPy / bilby / LAL convention; real-data PE recovers the
        published sky position with this sign.

        Note: this convention requires the waveform_fn passed to the sampler
        to NOT also apply ``tc`` internally. See
        :func:`gwjax.gwjax_samplers.build_ripplegw_waveform_fn` — it forces
        ``tc = 0`` in the ripplegw theta vector so the projection layer is
        the only place ``tc`` is applied (single-counted).

        Parameters
        ----------
        ra, dec : float [rad]  Source sky position.
        gmst    : float [rad]  Greenwich Mean Sidereal Time.

        Returns
        -------
        Δt : scalar JAX array [s]
        """
        cos_dec = jnp.cos(dec)
        sin_dec = jnp.sin(dec)
        ra_ecef = ra - gmst
        n_ecef = jnp.array([
            cos_dec * jnp.cos(ra_ecef),
            cos_dec * jnp.sin(ra_ecef),
            sin_dec,
        ])
        # Δt = t_det − t_geo = −(n̂ · vertex)/c
        # (Detector closer to source → n̂·vertex > 0 → Δt < 0 → signal earlier.)
        return -jnp.dot(self.vertex, n_ecef) / _C_LIGHT

    # ── Waveform projection ────────────────────────────────────────────────────

    def project_waveform(
        self,
        hp:         jnp.ndarray,
        hc:         jnp.ndarray,
        ra,
        dec,
        psi,
        gmst=0.0,
    ) -> jnp.ndarray:
        """Convenience: build (Fp, Fc, Δt) from sky parameters and project.

        Returns the detector FD strain ``h = (Fp·hp + Fc·hc)·exp(-2πif·Δt)``.
        All inputs may be traced; the call is jit/vmap-safe.
        """
        Fp, Fc = self.antenna_pattern(ra, dec, psi, gmst)
        dt     = self.time_delay_from_geocenter(ra, dec, gmst)
        return waveform_projection_fd(
            hp, hc, Fp, Fc, self.grid.frequency_domain_array, dt
        )

    # Re-export the pure functional form on the class for convenience.
    waveform_projection = staticmethod(waveform_projection_fd)

    def inject_signal(self, h_proj: jnp.ndarray, domain: str = "fd") -> None:
        """Add a projected waveform to the strain data (mutates in place).

        Parameters
        ----------
        h_proj : jax.Array
            Projected signal (complex FD or real TD).
        domain : ``"fd"`` or ``"td"``.

        Notes
        -----
        Only the array in the matching domain is updated; the opposite-domain
        array is left to be computed lazily by :meth:`strain_td` /
        :meth:`strain_fd` (kept here as a direct rfft/irfft call for
        backwards compatibility).
        """
        if self.grid is None:
            raise RuntimeError("Attach a TimeFrequencyGrid first via set_grid().")

        if domain == "fd":
            if self.strain_data_fd is None:
                raise RuntimeError("No FD data loaded to inject into.")
            self.strain_data_fd = self.strain_data_fd + h_proj
            self.strain_data_td = jnp.fft.irfft(
                self.strain_data_fd * self.grid.sampling_rate,
                n=self.grid.n_samples,
            ).real
        elif domain == "td":
            if self.strain_data_td is None:
                raise RuntimeError("No TD data loaded to inject into.")
            self.strain_data_td = self.strain_data_td + h_proj
            self.strain_data_fd = jnp.fft.rfft(self.strain_data_td) * self.grid.dt
        else:
            raise ValueError(f"domain must be 'fd' or 'td', got '{domain}'")

    # ── Likelihood / SNR (delegate to gwjax_likelihood_utils) ─────────────────

    log_likelihood = staticmethod(log_likelihood_ifo)

    @property
    def weight(self) -> jnp.ndarray:
        """Pre-computed inner-product weight ``w = 4·df·mask/psd``.

        Cached lazily; invalidated by :meth:`set_grid` and :meth:`set_psd`.
        Pass this directly to :func:`log_likelihood_ifo` /
        :func:`inner_product` / :func:`optimal_snr_squared` to skip the
        per-call ``1/psd`` and ``isfinite`` work.
        """
        if self._weight is None:
            if self.grid is None or self.psd is None:
                raise RuntimeError(
                    f"Interferometer '{self.name}' is missing grid/PSD; "
                    "cannot compute weight."
                )
            self._weight = precompute_weight(
                self.psd, self.grid.df, mask=self.grid.frequency_mask,
            )
        return self._weight

    def log_likelihood_data(self, h_fd: jnp.ndarray) -> jnp.ndarray:
        """Gaussian log-likelihood ``-0.5⟨d-h|d-h⟩`` against this IFO's data.

        Uses the cached inner-product :attr:`weight`.
        """
        if self.strain_data_fd is None:
            raise RuntimeError(f"Interferometer '{self.name}' has no data loaded.")
        return log_likelihood_ifo(self.strain_data_fd, h_fd, self.weight)

    def optimal_snr(self, h_fd: jnp.ndarray) -> jnp.ndarray:
        """Optimal SNR ρ = √⟨h|h⟩ for a candidate signal in this IFO."""
        return jnp.sqrt(optimal_snr_squared(h_fd, self.weight))

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        grid_info = (
            f"grid={self.grid.n_samples}samp@{self.grid.sampling_rate:.0f}Hz"
            if self.grid is not None else "no grid"
        )
        psd_info = "psd=set" if self.psd is not None else "psd=None"
        data_info = "data=loaded" if self.strain_data_td is not None else "data=None"
        return (
            f"Interferometer('{self.name}', L={self.arm_length:.0f}m, "
            f"{grid_info}, {psd_info}, {data_info})"
        )
