"""
gwjax_compatibility.py
=======================
Real-data ingest layer. Bridges between ``gwpy.TimeSeries`` objects and
gwjax :class:`~gwjax.gwjax_interferometer_utils.Interferometer` /
:class:`~gwjax.gwjax_network_ifos_utils.Network` instances.

Entry points
------------
  * :data:`KNOWN_EVENTS` — registry of well-known LVK events with their
    GPS times and detector availability.
  * :func:`fetch_event_strain` — pulls a segment around a registered event
    from GWOSC via ``gwpy.TimeSeries.fetch_open_data``.
  * :func:`read_strain_file` / :func:`read_strain_files` — load strain
    from one or several local files (GWF, HDF5, ASCII — anything
    ``gwpy.TimeSeries.read`` understands).
  * :func:`timeseries_to_ifo` — resample, crop, and install a TimeSeries
    into a single :class:`Interferometer`.
  * :func:`attach_data_to_network` — install one TimeSeries per detector
    into a :class:`Network`, optionally estimating PSDs from off-source
    data on the fly.
  * :func:`attach_event_to_network` — high-level convenience: fetch a
    known event + estimate PSDs + populate the whole network, with an
    optional ``deglitched_files`` override for BayesWave-cleaned strains
    (e.g. GW170817 L1).

Notes on the deglitched GW170817 strain
---------------------------------------
The GstLAL/BayesWave-cleaned L1 frame for GW170817 (Cornish & Littenberg
2015, DCC P1700407) is distributed separately by the LVK collaboration as
a ``.gwf`` file. ``gwpy.TimeSeries.fetch_open_data`` returns the *raw*
data. To use the cleaned strain, download the file (e.g. from
https://dcc.ligo.org/LIGO-T1700406/public) and pass its path to
:func:`attach_event_to_network` via ``deglitched_files={"L1": ...}``.

The module imports ``gwpy`` lazily, so the package can be imported
without ``gwpy`` installed; only the functions that touch real data
will fail at call time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import jax.numpy as jnp


# ── Greenwich Mean Sidereal Time at a GPS time ────────────────────────────────

def gmst_from_gps(gps_time: float) -> float:
    """Greenwich Mean Sidereal Time [rad] at a GPS epoch, mod 2π.

    Delegates to ``bilby_cython.time.greenwich_mean_sidereal_time`` — the
    IAU 2006 implementation used internally by bilby and LALSuite, with
    proper leap-second handling. Matches bilby to numerical precision
    (verified to < 1 µrad on the 50-point real-data sweep against bilby
    on GW150914 strain).

    The previous in-tree Aoki/Cornish-Littenberg polynomial used a wrong
    TAI-UTC count for post-2012 epochs (34 instead of 36 leap seconds at
    GW150914), introducing a ~200 µrad systematic GMST error that biased
    real-data PE by ~0.1 lnL across the sky. The bug was invisible to
    inject↔recover round-trips because both legs used the same wrong
    GMST.

    Parameters
    ----------
    gps_time : float
        Trigger GPS time [s].

    Returns
    -------
    gmst : float [rad]  in [0, 2π)
    """
    from bilby_cython.time import greenwich_mean_sidereal_time
    return float(greenwich_mean_sidereal_time(float(gps_time)) % (2.0 * math.pi))


# ── Lazy gwpy import ──────────────────────────────────────────────────────────

def _require_gwpy():
    """Import ``gwpy`` on demand and return the TimeSeries class."""
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "gwpy is required for real-data loading. Install with "
            "`pip install gwpy` (and `python-gwosc` for GWOSC access)."
        ) from e
    return TimeSeries


# ── Known-event registry ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class EventInfo:
    """Metadata for a publicly-released LVK event."""
    name:       str
    gps_time:   float          #: trigger time at geocenter [s]
    detectors:  tuple          #: detectors with public data
    bandpass:   tuple = (20.0, 500.0)  #: standard analysis band [Hz]
    notes:      str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EventInfo('{self.name}', gps={self.gps_time}, "
            f"detectors={self.detectors}, band={self.bandpass})"
        )


#: Reference GPS times taken from the GWOSC event catalogue.
KNOWN_EVENTS: dict = {
    "GW150914": EventInfo(
        name="GW150914",
        gps_time=1126259462.4,
        detectors=("H1", "L1"),
        bandpass=(20.0, 500.0),
        notes="First direct GW detection (BBH, ~35+30 Msun).",
    ),
    "GW151226": EventInfo(
        name="GW151226",
        gps_time=1135136350.6,
        detectors=("H1", "L1"),
        bandpass=(20.0, 500.0),
        notes="BBH, ~14+8 Msun.",
    ),
    "GW170104": EventInfo(
        name="GW170104",
        gps_time=1167559936.6,
        detectors=("H1", "L1"),
        bandpass=(20.0, 500.0),
    ),
    "GW170814": EventInfo(
        name="GW170814",
        gps_time=1186741861.5,
        detectors=("H1", "L1", "V1"),
        bandpass=(20.0, 500.0),
        notes="First three-detector BBH.",
    ),
    "GW170817": EventInfo(
        name="GW170817",
        gps_time=1187008882.4,
        detectors=("H1", "L1", "V1"),
        bandpass=(23.0, 2048.0),
        notes="First BNS detection. L1 contains a loud glitch ~1.1 s "
              "before merger; use the BayesWave-cleaned frame "
              "(DCC LIGO-T1700406) via deglitched_files={'L1': ...}.",
    ),
    "GW190521": EventInfo(
        name="GW190521",
        gps_time=1242442967.4,
        detectors=("H1", "L1", "V1"),
        bandpass=(20.0, 500.0),
        notes="Heaviest BBH confidently detected to date.",
    ),
}


def list_events() -> list:
    """Return sorted list of registered event names."""
    return sorted(KNOWN_EVENTS)


def get_event_info(event: str) -> EventInfo:
    """Look up an :class:`EventInfo` by name."""
    if event not in KNOWN_EVENTS:
        raise KeyError(
            f"Unknown event '{event}'. Registered events: {list_events()}"
        )
    return KNOWN_EVENTS[event]


# ── GWOSC fetch ───────────────────────────────────────────────────────────────

def fetch_event_strain(
    event:          str,
    detectors:      Optional[list] = None,
    duration:       float = 4.0,
    sampling_rate:  float = 4096.0,
    pre_merger:     Optional[float] = None,
    cache:          bool = True,
    verbose:        bool = False,
) -> dict:
    """Download a strain segment around a registered event from GWOSC.

    Parameters
    ----------
    event : str
        Registered event name (see :func:`list_events`).
    detectors : list of str, optional
        Detector names to fetch. Defaults to the event's full detector
        list. Anything not in the event's registered detector list is
        skipped with a warning.
    duration : float
        Segment duration [s].
    sampling_rate : float
        Target sample rate [Hz]. ``gwpy`` returns whatever rate is
        available on GWOSC closest to this value.
    pre_merger : float, optional
        Seconds of pre-merger data to include in the segment. Defaults
        to ``0.875 * duration`` (merger sits 7/8 through). Must satisfy
        ``0 < pre_merger < duration``.
    cache : bool
        Cache the downloaded frame in gwpy's default cache directory.
    verbose : bool
        Pass-through to gwpy's progress bar.

    Returns
    -------
    dict[str, gwpy.timeseries.TimeSeries]
        Cropped (and resampled if necessary) strains keyed by detector.
    """
    TimeSeries = _require_gwpy()
    info = get_event_info(event)

    requested = list(detectors) if detectors is not None else list(info.detectors)
    missing = [d for d in requested if d not in info.detectors]
    if missing:
        import warnings
        warnings.warn(
            f"Detectors {missing} have no public {event} data; skipping.",
            stacklevel=2,
        )
    targets = [d for d in requested if d in info.detectors]
    if not targets:
        raise ValueError(
            f"No requested detectors are available for {event}. "
            f"Available: {info.detectors}"
        )

    if pre_merger is None:
        pre_merger = 0.875 * duration
    if not (0.0 < pre_merger < duration):
        raise ValueError(
            f"pre_merger ({pre_merger}) must lie in (0, duration={duration})."
        )

    seg_start = info.gps_time - pre_merger
    seg_end   = seg_start + duration

    out = {}
    for det in targets:
        ts = TimeSeries.fetch_open_data(
            det, seg_start, seg_end,
            sample_rate=int(sampling_rate),
            cache=cache, verbose=verbose,
        )
        out[det] = ts
    return out


# ── Local-file loaders ────────────────────────────────────────────────────────

def read_strain_file(
    filepath:  Union[str, Path],
    channel:   Optional[str] = None,
    fmt:       Optional[str] = None,
    **kwargs,
):
    """Load a single strain TimeSeries from a local file.

    Parameters
    ----------
    filepath : str or Path
        Path to a GWF/HDF5/ASCII/CSV file readable by ``gwpy``.
    channel : str, optional
        Channel name (required for GWF and most HDF5 layouts).
    fmt : str, optional
        Force a specific format. Examples: ``"gwf"``, ``"hdf5.gwosc"``,
        ``"txt"``. If omitted, ``gwpy`` infers from the extension.
    **kwargs
        Forwarded to :meth:`gwpy.timeseries.TimeSeries.read`.

    Returns
    -------
    gwpy.timeseries.TimeSeries
    """
    TimeSeries = _require_gwpy()
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    read_kwargs = dict(kwargs)
    if channel is not None:
        read_kwargs["channel"] = channel
    if fmt is not None:
        read_kwargs["format"] = fmt
    return TimeSeries.read(str(filepath), **read_kwargs)


def read_strain_files(
    files:    dict,
    channels: Optional[dict] = None,
    fmt:      Optional[str] = None,
    **kwargs,
) -> dict:
    """Load multiple strain TimeSeries from a mapping of detector -> filepath.

    Parameters
    ----------
    files : dict[str, str|Path]
        Mapping ``{'H1': 'path/H-H1.gwf', 'L1': 'path/L-L1.gwf', ...}``.
    channels : dict[str, str], optional
        Per-detector channel names. If a key is missing, no channel is set.
    fmt : str, optional
        Format applied to every file.

    Returns
    -------
    dict[str, gwpy.timeseries.TimeSeries]
    """
    channels = channels or {}
    return {
        name: read_strain_file(
            fp, channel=channels.get(name), fmt=fmt, **kwargs,
        )
        for name, fp in files.items()
    }


# ── TimeSeries → Interferometer ───────────────────────────────────────────────

def timeseries_to_ifo(
    timeseries,
    ifo,
    event_gps:    Optional[float] = None,
    pre_merger:   Optional[float] = None,
    bandpass:     Optional[tuple] = None,
    detrend:      bool = True,
):
    """Resample, crop, and install a TimeSeries into an Interferometer.

    Parameters
    ----------
    timeseries : gwpy.timeseries.TimeSeries
    ifo : Interferometer
        Must have a :class:`TimeFrequencyGrid` already set.
    event_gps : float, optional
        GPS time of the merger/trigger. If given, the segment is
        positioned so the event sits ``pre_merger`` seconds from the
        segment start. If ``None`` the TimeSeries is cropped from
        ``ifo.grid.initial_time``.
    pre_merger : float, optional
        Seconds of pre-merger data. Defaults to ``0.875 * duration``.
        Ignored when ``event_gps`` is None.
    bandpass : (low, high), optional
        Apply a Butterworth bandpass before cropping.
    detrend : bool
        Subtract the segment mean before installing.

    Returns
    -------
    gwpy.timeseries.TimeSeries
        The cropped/resampled series actually installed (useful for
        further QA — plotting, whitening, etc.).
    """
    if ifo.grid is None:
        raise RuntimeError(
            f"Interferometer '{ifo.name}' has no TimeFrequencyGrid. "
            "Call set_grid() first."
        )

    grid = ifo.grid
    fs   = grid.sampling_rate
    N    = grid.n_samples

    # Resample if needed
    if abs(float(timeseries.sample_rate.value) - fs) > 1e-6:
        timeseries = timeseries.resample(fs)

    # Bandpass (helps avoid aliasing on the segment edges after cropping)
    if bandpass is not None:
        timeseries = timeseries.bandpass(bandpass[0], bandpass[1])

    # Determine crop window
    if event_gps is not None:
        if pre_merger is None:
            pre_merger = 0.875 * grid.duration
        if not (0.0 < pre_merger < grid.duration):
            raise ValueError(
                f"pre_merger ({pre_merger}) must lie in (0, duration={grid.duration})."
            )
        seg_start = event_gps - pre_merger
    else:
        seg_start = grid.initial_time
    seg_end = seg_start + grid.duration

    cropped = timeseries.crop(seg_start, seg_end)

    # Enforce exact length (gwpy may round)
    arr = np.asarray(cropped.value, dtype=np.float64)
    if arr.size != N:
        if arr.size > N:
            arr = arr[:N]
        else:
            raise ValueError(
                f"TimeSeries for {ifo.name} only has {arr.size} samples "
                f"after cropping; grid requires {N}. Fetch a longer segment."
            )

    if detrend:
        arr = arr - arr.mean()

    ifo.load_data(jnp.asarray(arr), domain="td")
    return cropped


# ── Network-level attachment ──────────────────────────────────────────────────

def attach_data_to_network(
    network,
    timeseries_dict:        dict,
    event_gps:              Optional[float] = None,
    pre_merger:             Optional[float] = None,
    bandpass:               Optional[tuple] = None,
    estimate_psd:           bool = False,
    psd_segment_duration:   float = 32.0,
    psd_offset:             float = 8.0,
    detrend:                bool = True,
) -> list:
    """Install per-detector strains into every matching IFO of a network.

    Parameters
    ----------
    network : Network
    timeseries_dict : dict[str, gwpy.timeseries.TimeSeries]
    event_gps, pre_merger, bandpass, detrend
        Forwarded to :func:`timeseries_to_ifo`.
    estimate_psd : bool
        If True, estimate each IFO's PSD via Welch on a quiet stretch
        preceding the on-source segment.
    psd_segment_duration : float
        Duration [s] of the off-source stretch used for the Welch
        estimate. Requires the input TimeSeries to extend at least
        ``psd_segment_duration + psd_offset`` seconds before the event.
    psd_offset : float
        Gap [s] between the off-source PSD segment and the on-source
        segment. A nonzero gap prevents the merger leaking into the PSD.

    Returns
    -------
    list[Interferometer]
        IFOs that were successfully populated.
    """
    from gwjax.gwjax_psd_utils import psd_from_data  # local import: avoid cycle

    populated = []
    for ifo in network.interferometers:
        ts = timeseries_dict.get(ifo.name)
        if ts is None:
            continue
        timeseries_to_ifo(
            ts, ifo,
            event_gps=event_gps,
            pre_merger=pre_merger,
            bandpass=bandpass,
            detrend=detrend,
        )

        if estimate_psd:
            if event_gps is None:
                raise ValueError(
                    "estimate_psd=True requires event_gps so the off-source "
                    "window can be placed."
                )
            off_end   = event_gps - psd_offset
            off_start = off_end - psd_segment_duration
            try:
                off = ts.crop(off_start, off_end)
            except Exception as e:
                raise ValueError(
                    f"Off-source PSD window [{off_start}, {off_end}] is "
                    f"outside the TimeSeries for {ifo.name}. Fetch more "
                    "data or shrink psd_segment_duration/psd_offset."
                ) from e
            if abs(float(off.sample_rate.value) - ifo.grid.sampling_rate) > 1e-6:
                off = off.resample(ifo.grid.sampling_rate)
            _, psd = psd_from_data(
                np.asarray(off.value, dtype=np.float64),
                sampling_rate=ifo.grid.sampling_rate,
                target_freqs=ifo.grid.frequency_domain_array,
            )
            ifo.psd = jnp.asarray(psd)

        populated.append(ifo)

    if not populated:
        raise ValueError(
            "None of the network IFOs matched the supplied timeseries_dict. "
            f"Network: {[i.name for i in network.interferometers]}; "
            f"Data:    {list(timeseries_dict)}"
        )
    return populated


def attach_event_to_network(
    network,
    event:                  str,
    estimate_psd:           bool = True,
    psd_segment_duration:   float = 32.0,
    psd_offset:             float = 8.0,
    bandpass:               Optional[tuple] = None,
    deglitched_files:       Optional[dict] = None,
    cache:                  bool = True,
    verbose:                bool = False,
) -> list:
    """Fetch a known event from GWOSC (plus optional deglitched files)
    and populate every matching detector in ``network``.

    The total fetch window per detector is

        duration + psd_segment_duration + 2 · psd_offset,

    centered on the event so both the on-source segment and the
    off-source PSD window fit inside.

    Parameters
    ----------
    network : Network
        Must already have a :class:`TimeFrequencyGrid` attached to every IFO.
    event : str
        Registered event name (see :func:`list_events`).
    estimate_psd : bool
        Estimate each detector's PSD from the off-source segment.
    psd_segment_duration, psd_offset : float
        See :func:`attach_data_to_network`.
    bandpass : (low, high), optional
        Bandpass to apply to all strains before storage. Defaults to the
        event's standard analysis band from :data:`KNOWN_EVENTS`.
    deglitched_files : dict[str, str|Path], optional
        Per-detector override files (e.g. BayesWave-cleaned frames for
        GW170817 L1). Files are read via :func:`read_strain_file` and
        replace the GWOSC fetch for those detectors. The off-source PSD
        is then estimated from the deglitched stretch as well.
    cache, verbose
        Forwarded to :func:`fetch_event_strain`.

    Returns
    -------
    list[Interferometer]
        IFOs that were successfully populated.
    """
    info = get_event_info(event)

    matching = [ifo for ifo in network.interferometers
                if ifo.name in info.detectors]
    if not matching:
        raise ValueError(
            f"No network IFOs match {event}'s available detectors "
            f"{info.detectors}. Network has "
            f"{[i.name for i in network.interferometers]}."
        )

    grid = matching[0].grid
    if grid is None:
        raise RuntimeError(
            f"Interferometer '{matching[0].name}' has no grid attached."
        )

    if bandpass is None:
        bandpass = info.bandpass

    # GWOSC fetch (covers on-source + off-source + safety margin)
    half_total = grid.duration + psd_segment_duration + 2.0 * psd_offset
    pre_merger = 0.875 * grid.duration + psd_segment_duration + psd_offset

    deglitched_files = dict(deglitched_files or {})
    gwosc_targets = [ifo.name for ifo in matching
                     if ifo.name not in deglitched_files]

    timeseries_dict = {}
    if gwosc_targets:
        TimeSeries = _require_gwpy()
        seg_start = info.gps_time - pre_merger
        seg_end   = seg_start + half_total
        for det in gwosc_targets:
            timeseries_dict[det] = TimeSeries.fetch_open_data(
                det, seg_start, seg_end,
                sample_rate=int(grid.sampling_rate),
                cache=cache, verbose=verbose,
            )

    # Local deglitched overrides
    for det, fp in deglitched_files.items():
        timeseries_dict[det] = read_strain_file(fp)

    populated = attach_data_to_network(
        network,
        timeseries_dict,
        event_gps=info.gps_time,
        pre_merger=0.875 * grid.duration,
        bandpass=bandpass,
        estimate_psd=estimate_psd,
        psd_segment_duration=psd_segment_duration,
        psd_offset=psd_offset,
    )

    # Write GMST(trigger) onto the network so the sampler defaults its
    # antenna-pattern + time-delay calculations to the **celestial** frame
    # (matching bilby/LAL/SHARPy). Without this, ``ra`` would live in an
    # Earth-rotating ECEF frame and be offset from the catalogued J2000 RA
    # by exactly the trigger GMST.
    network.gmst = gmst_from_gps(info.gps_time)

    return populated
