"""bilby frequency-domain source models for TEOBResumSPA (TEOBResumS-GIOTTO, SPA Fourier domain).

TEOBResumSPA is the model mlgw_bns_jax's default surrogate is trained on (mlgw_bns
TEOBResumSGenerator: EOBRunPy with ``domain = 1``) and the BNS baseline the referee names. It is
not a LALSimulation approximant, so it is called through the ``teobresums`` Python package
(``pip install teobresums``; module ``EOBRun_module``).

Conventions (checked on the 128 s / 23-2048 Hz grid, see README):
  * hp = Re(h+) - i Im(h+) of the EOBRunPy FD output, same for hx -- the convention mlgw_bns uses;
    hx/hp and |hp| then agree with LAL (IMRPhenomD_NRTidalv2: same time/phase shift aligns hp and
    hx to overlap 0.99985; |hp| ratio 0.9998).
  * mlgw_bns_jax reproduces it to overlap 0.999999, with the template time origin 0.81 ms earlier
    (TEOB features occur 0.81 ms later for the same tc). tc is marginalised/sampled over a window,
    so only the window centre depends on this; run_bilby_gw170817.py centres it on the peak.
  * (2,2) mode only (``use_mode_lm = [1]``), aligned spins, q = m1/m2 >= 1 with chi1, Lambda_A
    belonging to the heavier star -- bilby's mass_1 >= mass_2 ordering.
"""
import numpy as np

_EOB = None


def _eobrun():
    global _EOB
    if _EOB is None:
        from EOBRun_module import EOBRunPy
        _EOB = EOBRunPy
    return _EOB


def teobresumspa_polarizations(freqs, mass_1, mass_2, chi_1, chi_2, lambda_1, lambda_2,
                               luminosity_distance, theta_jn, phase, f_start):
    """(hp, hx) of TEOBResumSPA at the frequencies ``freqs`` [Hz] (all >= f_start)."""
    pars = dict(
        M=mass_1 + mass_2, q=mass_1 / mass_2, chi1=chi_1, chi2=chi_2,
        LambdaAl2=lambda_1, LambdaBl2=lambda_2, distance=luminosity_distance,
        inclination=theta_jn, coalescence_angle=phase, initial_frequency=f_start,
        use_geometric_units="no", domain=1, interp_freqs="yes", freqs=list(map(float, freqs)),
        srate_interp=4096.0, output_hpc="no", time_shift_FD="yes", ode_tmax=1e12,
        use_mode_lm=[1], arg_out="no")
    _, rhp, ihp, rhc, ihc = _eobrun()(pars)[:5]
    return np.asarray(rhp) - 1j * np.asarray(ihp), np.asarray(rhc) - 1j * np.asarray(ihc)


def _start(minimum_frequency):
    # integrate from a little below f_min (mlgw_bns starts 256 df early for the same reason:
    # TEOB's phase needs a few cycles to settle after the initial-data transient)
    return 0.9 * minimum_frequency


def teobresumspa_bns(frequency_array, mass_1, mass_2, luminosity_distance, chi_1, chi_2,
                     lambda_1, lambda_2, theta_jn, phase, **kwargs):
    """bilby FD source model on the full grid (zeros outside [minimum, maximum]_frequency)."""
    f_min = kwargs.get("minimum_frequency", 20.0)
    f_max = kwargs.get("maximum_frequency", frequency_array[-1])
    band = (frequency_array >= f_min) & (frequency_array <= f_max)
    hp = np.zeros(len(frequency_array), complex)
    hx = np.zeros(len(frequency_array), complex)
    hp[band], hx[band] = teobresumspa_polarizations(
        frequency_array[band], mass_1, mass_2, chi_1, chi_2, lambda_1, lambda_2,
        luminosity_distance, theta_jn, phase, _start(f_min))
    return dict(plus=hp, cross=hx)


def teobresumspa_bns_relative_binning(frequency_array, mass_1, mass_2, luminosity_distance,
                                      chi_1, chi_2, lambda_1, lambda_2, theta_jn, phase, **kwargs):
    """Source model for bilby's RelativeBinningGravitationalWaveTransient.

    ``fiducial = 1``: full grid (fiducial waveform); otherwise only at ``frequency_bin_edges``.
    bilby versions differ in how they flag the fiducial call (2.8.2 sets waveform_arguments
    ["fiducial"] = 1; the igwn env on LDG did not), so a call without bin edges is also treated as
    the fiducial (full-grid) call -- bin edges only exist once the binning is set up.
    """
    fiducial = kwargs.pop("fiducial", 0)
    if fiducial == 1 or kwargs.get("frequency_bin_edges") is None:
        return teobresumspa_bns(frequency_array, mass_1, mass_2, luminosity_distance, chi_1, chi_2,
                                lambda_1, lambda_2, theta_jn, phase, **kwargs)
    edges = np.asarray(kwargs["frequency_bin_edges"])
    hp, hx = teobresumspa_polarizations(
        edges, mass_1, mass_2, chi_1, chi_2, lambda_1, lambda_2, luminosity_distance, theta_jn,
        phase, _start(min(edges[0], kwargs.get("minimum_frequency", edges[0]))))
    return dict(plus=hp, cross=hx)
