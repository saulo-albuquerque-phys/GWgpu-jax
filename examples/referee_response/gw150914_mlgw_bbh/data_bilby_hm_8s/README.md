# GW150914, 8 s segment: inputs and bilby reference of the 7-mode SEOBNRv5HM run

Copied verbatim from the earlier bilby/dynesty analysis (bilby 2.8.2, `gwsignal_binary_black_hole`,
SEOBNRv5HM with modes (2,2) (2,1) (3,3) (3,2) (4,4) (4,3) (5,5), nlive 1000, naccept 60, dlogz 0.1,
phase sampled, nothing marginalised; 97.8 h on 16 cores, 4.73e7 likelihood calls).

* `gw150914_data.npz` — the conditioned data that run analysed: `data_fd`, `psd` (H1, L1), `freqs`;
  8 s at 2048 Hz, trigger at `t_ref` = 6 s (`gps` = 1126259462.4), band 20–1000 Hz, PSD = median Welch
  over 512 s, Tukey roll-off 0.4 s. GWOSC open data.
* `bilby_SEOBNRv5HM_7modes_posterior.npz` — its posterior: `samples` (5107 x 10) with `names`
  mtot, q (= m1/m2), chi1, chi2, d_l, cos_iota, psi, ra, sin_dec, t_c (s from segment start);
  `log_z` (= ln Bayes factor, 261.85 +- 0.21), `log_noise_evidence`.

Priors: uniform M_tot in [55, 120], uniform q = m1/m2 in [1, 8], uniform chi_1,2 in [-0.8, 0.8],
d_L ~ d^2 on [100, 3000] Mpc, isotropic orientation and sky, psi in [0, pi], t_c within +-0.05 s of 6 s.

Used by `COLAB_GW150914_mlgw_bbh_HM_vs_bilby8s.ipynb` (built by `build_notebook_hm_bilby8s.py`).
