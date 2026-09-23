# neurophysiological-prior-network# NPN: Neurophysiologically-Primed Network for Motor-Imagery EEG Decoding

Code for the paper **"Quantifying the Contribution of Neurophysiological Priors
to Deep Motor-Imagery Decoding: Channel Dependence and Data Efficiency."**

NPN is a compact multi-band convolutional decoder whose per-band spatial layer
is initialized from Common Spatial Pattern (CSP) filters and refined end to end.
This repository reproduces the paper's controlled experiments: an ablation
against random initialization, a channel-count intervention, a data-efficiency
curve, a CSP-frozen control, and a four-dataset benchmark against classical and
Riemannian baselines with hierarchical statistics.

## Install

```bash
git clone https://github.com/<you>/npn.git
cd npn
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.9+, PyTorch, MNE, pyriemann, scikit-learn, SciPy, NumPy,
pandas, matplotlib.

## Datasets

The four public datasets are not redistributed here. Download them and fill in
the four adapter functions in `npn/loaders.py`; each returns epoched trials
`X` of shape `(n_trials, n_channels, n_samples)` band-pass filtered to 8-30 Hz
and integer labels `y`.

| Dataset | Subjects | Classes | Channels | fs (Hz) | Role |
|---|---|---|---|---|---|
| BCI-IV-2a | 9 | 4 | 22 | 250 | Primary controlled experiments |
| BCI-IV-2b | 9 | 2 | 3 | 250 | External 3-channel consistency |
| PhysioNet EEGMMIDB | 20 | 2 | 17 | 160 | Benchmark |
| High-Gamma | 14 | 2 | 17 | 250 | Benchmark |

## Reproduce the experiments

```bash
# four-dataset benchmark (NPN, CSP, FBCSP, MDM, TS+LR)
python scripts/run_experiments.py benchmark  --datasets 2a,2b,PhysioNet,HGD --root /path/to/data

# channel-count intervention on 2a (prior gain vs number of channels)
python scripts/run_experiments.py channels   --root /path/to/data

# data-efficiency curve on 2a and 2b
python scripts/run_experiments.py efficiency --datasets 2a,2b --root /path/to/data

# random / CSP-init / CSP-frozen control on 2a
python scripts/run_experiments.py frozen     --root /path/to/data

# hierarchical bootstrap + Wilcoxon from results/benchmark.csv
python scripts/run_experiments.py stats
```

Each subcommand writes a CSV under `results/` and is crash-safe: finished cells
are skipped, so a run can be interrupted and resumed. Deep baselines
(EEGNet, ShallowConvNet, DeepConvNet, EEG-Conformer, ATCNet, EEG-ITNet,
EEG-Inception, EEG-Inception-MI, TCN) are trained with
[braindecode](https://braindecode.org/) under the same protocol and are not
included in this driver.

## Figures

```bash
python scripts/make_figures.py --results results --out figures
```

Produces `fig_channel_scaling`, `fig_data_efficiency`, `fig_mean_rank`, and
`fig_main_accuracy` as 300 dpi PDF and PNG. The proposed model is labeled NPN.

## Package layout

```
npn/
  model.py       NPN architecture (multi-band CSP-primed conv net)
  csp.py         CSP filters (GEVD + whitening, Ledoit-Wolf, one-vs-rest)
  data.py        band decomposition, augmentation, channel subsampling
  train.py       training loop, k-fold CV, csp/random/frozen conditions
  baselines.py   CSP+LDA, MDM, tangent-space logistic regression
  stats.py       hierarchical bootstrap, Wilcoxon-Holm, Cohen's dz
  loaders.py     dataset adapters (fill these in) + exact 2a channel sets
scripts/
  run_experiments.py   benchmark / channels / efficiency / frozen / stats
  make_figures.py      regenerate the paper figures
tests/
  test_npn.py    unit tests (run with pytest)
```

## Method summary

- The per-band spatial layer is initialized from CSP filters and trained end to
  end; a squared-response average pool is a differentiable estimator of band
  power (the learned counterpart of the CSP log-variance feature).
- CSP is estimated on the training partition of each fold only (no leakage).
- The channel-count and data-efficiency interventions hold every other factor
  fixed and change only the montage size or the training fraction, so the
  measured effect is attributable to that factor.

## Citation

```bibtex
@article{npn2026,
  title   = {Quantifying the Contribution of Neurophysiological Priors to Deep
             Motor-Imagery Decoding: Channel Dependence and Data Efficiency},
  author  = {Tiwari, Anurag and others},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
