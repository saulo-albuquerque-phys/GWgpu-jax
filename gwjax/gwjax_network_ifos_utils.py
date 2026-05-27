"""
gwjax_network_ifos_utils.py
============================
Multi-detector interferometer network.

:class:`Network` combines two or more :class:`~gwjax.gwjax_interferometer_utils.Interferometer`
instances that share the same :class:`~gwjax.gwjax_timefrequencydomain_utils.TimeFrequencyGrid`.
It provides convenience methods for joint noise generation and antenna-pattern
evaluation across the whole network.

Usage
-----
>>> from gwjax.gwjax_timefrequencydomain_utils import TimeFrequencyGrid
>>> from gwjax.gwjax_interferometer_utils import Interferometer
>>> from gwjax.gwjax_network_ifos_utils import Network
>>>
>>> grid = TimeFrequencyGrid(duration=4.0, sampling_rate=2048.0)
>>>
>>> # Construct from pre-built Interferometer objects
>>> h1 = Interferometer("H1", grid=grid)
>>> l1 = Interferometer("L1", grid=grid)
>>> net = Network([h1, l1])
>>>
>>> # Or use the factory class-method (constructs the IFOs automatically)
>>> net = Network.from_names(["H1", "L1", "V1"], grid)
>>>
>>> # Generate independent noise realisations
>>> net.generate_noise(seed=0)
>>>
>>> # Antenna patterns at a given sky position
>>> import jax.numpy as jnp
>>> fp, fc = net.antenna_patterns(ra=1.37, dec=-1.22, psi=0.5, gmst=0.0)
>>> # fp, fc are dicts keyed by detector name

The network also exposes ``time_delays`` and convenience properties
(``names``, ``interferometers``, ``n_detectors``).
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from gwjax.gwjax_timefrequencydomain_utils import TimeFrequencyGrid
from gwjax.gwjax_interferometer_utils import Interferometer
from gwjax.gwjax_likelihood_utils import (
    log_likelihood_ifo,
    log_likelihood_network,
    optimal_snr_squared,
)


class Network:
    """Collection of gravitational-wave interferometers sharing a common grid.

    Parameters
    ----------
    interferometers : sequence of :class:`Interferometer`
        At least two detectors.  They should all share the same
        :class:`TimeFrequencyGrid`; a warning is issued if grids differ.

    Attributes
    ----------
    interferometers : list of Interferometer
    names : list of str
    n_detectors : int
    """

    def __init__(self, interferometers: Sequence[Interferometer]) -> None:
        ifos = list(interferometers)
        if len(ifos) < 2:
            raise ValueError(
                f"A network requires at least 2 interferometers, got {len(ifos)}."
            )
        # Soft check: warn if grids differ
        grids = [ifo.grid for ifo in ifos if ifo.grid is not None]
        if len(grids) > 1:
            ref = grids[0]
            for g in grids[1:]:
                if (g.n_samples != ref.n_samples
                        or g.sampling_rate != ref.sampling_rate):
                    import warnings
                    warnings.warn(
                        "Not all interferometers share the same grid. "
                        "Make sure the grids are compatible.",
                        stacklevel=2,
                    )
                    break

        self._ifos: list[Interferometer] = ifos

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_names(
        cls,
        names:      Sequence[str],
        grid:       TimeFrequencyGrid,
        psd_models: dict[str, str | jnp.ndarray] | None = None,
    ) -> "Network":
        """Construct a network by name, sharing a common grid.

        Parameters
        ----------
        names : sequence of str
            Detector names, e.g. ``["H1", "L1", "V1"]``.
        grid : TimeFrequencyGrid
            Shared time/frequency grid.
        psd_models : dict, optional
            Mapping from detector name to PSD model override.
            Detectors not listed use their default model.

        Returns
        -------
        Network
        """
        psd_models = psd_models or {}
        ifos = [
            Interferometer(name, grid=grid, psd_model=psd_models.get(name))
            for name in names
        ]
        return cls(ifos)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def interferometers(self) -> list[Interferometer]:
        return self._ifos

    @property
    def names(self) -> list[str]:
        return [ifo.name for ifo in self._ifos]

    @property
    def n_detectors(self) -> int:
        return len(self._ifos)

    def __getitem__(self, key: str | int) -> Interferometer:
        if isinstance(key, str):
            for ifo in self._ifos:
                if ifo.name == key:
                    return ifo
            raise KeyError(f"No interferometer named '{key}' in network.")
        return self._ifos[key]

    # ── Noise generation ──────────────────────────────────────────────────────

    def generate_noise(
        self,
        seed: int | jnp.ndarray = 0,
    ) -> None:
        """Generate independent coloured noise for every interferometer.

        Each detector receives an independent JAX random key derived by
        splitting the master key *n_detectors* times, ensuring that noise
        realisations are statistically independent across the network.

        Parameters
        ----------
        seed : int or JAX PRNGKey
            Master random seed.
        """
        key = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
        keys = jax.random.split(key, self.n_detectors)
        for ifo, k in zip(self._ifos, keys):
            ifo.generate_noise(seed=k)

    # ── Antenna patterns ──────────────────────────────────────────────────────

    def antenna_patterns(
        self,
        ra:   float | jnp.ndarray,
        dec:  float | jnp.ndarray,
        psi:  float | jnp.ndarray,
        gmst: float | jnp.ndarray = 0.0,
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        """Compute F₊ and F× for every detector in the network.

        Parameters
        ----------
        ra, dec, psi, gmst : float [rad]
            Source sky position, polarisation angle, and GMST.

        Returns
        -------
        Fp : dict[str, scalar JAX array]
        Fc : dict[str, scalar JAX array]
            Keyed by detector name.
        """
        Fp, Fc = {}, {}
        for ifo in self._ifos:
            fp, fc = ifo.antenna_pattern(ra, dec, psi, gmst)
            Fp[ifo.name] = fp
            Fc[ifo.name] = fc
        return Fp, Fc

    # ── Time delays ───────────────────────────────────────────────────────────

    def time_delays(
        self,
        ra:   float | jnp.ndarray,
        dec:  float | jnp.ndarray,
        gmst: float | jnp.ndarray = 0.0,
    ) -> dict[str, jnp.ndarray]:
        """Arrival-time delay relative to the geocenter for each detector [s].

        Parameters
        ----------
        ra, dec : float [rad]  Source sky position.
        gmst    : float [rad]  Greenwich Mean Sidereal Time.

        Returns
        -------
        dict[str, scalar JAX array]  Keyed by detector name.
        """
        return {
            ifo.name: ifo.time_delay_from_geocenter(ra, dec, gmst)
            for ifo in self._ifos
        }

    # ── Data access helpers ───────────────────────────────────────────────────

    @property
    def strain_data_fd(self) -> dict[str, jnp.ndarray | None]:
        """Dict of FD strain arrays keyed by detector name."""
        return {ifo.name: ifo.strain_data_fd for ifo in self._ifos}

    @property
    def strain_data_td(self) -> dict[str, jnp.ndarray | None]:
        """Dict of TD strain arrays keyed by detector name."""
        return {ifo.name: ifo.strain_data_td for ifo in self._ifos}

    @property
    def psds(self) -> dict[str, jnp.ndarray | None]:
        """Dict of PSD arrays keyed by detector name."""
        return {ifo.name: ifo.psd for ifo in self._ifos}

    # ── Representation ────────────────────────────────────────────────────────

    # ── Waveform projection and injection ──────────────────────────────────────

    def project_waveform(
        self,
        hp:  jnp.ndarray,
        hc:  jnp.ndarray,
        ra,
        dec,
        psi,
        gmst=0.0,
    ) -> dict:
        """Project the same (h+, h×) onto every detector.

        For each IFO: ``h_i = (Fp_i·hp + Fc_i·hc) · exp(-2πi·f·Δt_i)``.

        Returns
        -------
        dict[str, jnp.ndarray]  Per-detector FD strain.
        """
        return {ifo.name: ifo.project_waveform(hp, hc, ra, dec, psi, gmst)
                for ifo in self._ifos}

    def inject_signal(
        self,
        h_proj_dict: dict[str, jnp.ndarray],
        domain: str = "fd",
    ) -> None:
        """Inject projected waveforms into all interferometers' data.

        Parameters
        ----------
        h_proj_dict : dict[str, jnp.ndarray]
            Keyed by detector name; values are projected waveforms
            (complex for FD, real for TD).
        domain : ``"fd"`` or ``"td"``
            Domain of the injected signal.
        """
        for ifo in self._ifos:
            if ifo.name in h_proj_dict:
                ifo.inject_signal(h_proj_dict[ifo.name], domain=domain)

    # ── Likelihood / SNR ──────────────────────────────────────────────────────

    # Re-export the pure functional form on the class for convenience.
    log_likelihood_single_ifo = staticmethod(log_likelihood_ifo)

    def _gather_data(self):
        """Collect per-IFO data and pre-computed weights into matching dicts.

        Raises if any detector is missing grid, PSD, or data.
        """
        d_fd_dict, weight_dict = {}, {}
        for ifo in self._ifos:
            if ifo.grid is None:
                raise RuntimeError(f"Interferometer '{ifo.name}' has no grid.")
            if ifo.psd is None:
                raise RuntimeError(f"Interferometer '{ifo.name}' has no PSD set.")
            if ifo.strain_data_fd is None:
                raise RuntimeError(f"Interferometer '{ifo.name}' has no data.")
            d_fd_dict[ifo.name]   = ifo.strain_data_fd
            weight_dict[ifo.name] = ifo.weight  # cached
        return d_fd_dict, weight_dict

    def log_likelihood_network(
        self,
        h_proj_dict: dict,
        domain: str = "fd",
    ) -> jnp.ndarray:
        """Compute joint log-likelihood over all interferometers.

        Parameters
        ----------
        h_proj_dict : dict[str, jnp.ndarray]
            Keyed by detector name; values are projected waveforms. Any
            detectors missing from the dict are treated as having zero signal.
        domain : ``"fd"`` or ``"td"``
            Domain of the supplied signals. TD signals are converted to FD
            via the IFO's grid before the inner product.

        Returns
        -------
        logl : scalar JAX array
        """
        d_fd_dict, weight_dict = self._gather_data()

        h_fd_dict = {}
        for ifo in self._ifos:
            sig = h_proj_dict.get(ifo.name)
            if sig is None:
                h_fd_dict[ifo.name] = jnp.zeros_like(ifo.strain_data_fd)
            elif domain == "fd":
                h_fd_dict[ifo.name] = sig
            elif domain == "td":
                h_fd_dict[ifo.name] = jnp.fft.rfft(sig) * ifo.grid.dt
            else:
                raise ValueError(f"domain must be 'fd' or 'td', got '{domain}'")

        return log_likelihood_network(d_fd_dict, h_fd_dict, weight_dict)

    def optimal_snr(
        self,
        h_proj_dict: dict,
    ) -> dict:
        """Per-detector optimal SNR ρ = √⟨h|h⟩ for a candidate signal."""
        snrs = {}
        for ifo in self._ifos:
            if ifo.name not in h_proj_dict:
                continue
            snrs[ifo.name] = jnp.sqrt(
                optimal_snr_squared(h_proj_dict[ifo.name], ifo.weight)
            )
        return snrs

    def network_optimal_snr(self, h_proj_dict: dict) -> jnp.ndarray:
        """Quadrature sum √Σ ρ_i² across detectors present in ``h_proj_dict``."""
        rho_sq = jnp.asarray(0.0)
        for ifo in self._ifos:
            if ifo.name not in h_proj_dict:
                continue
            rho_sq = rho_sq + optimal_snr_squared(
                h_proj_dict[ifo.name], ifo.weight,
            )
        return jnp.sqrt(rho_sq)

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        names = ", ".join(f"'{n}'" for n in self.names)
        return f"Network([{names}])"

