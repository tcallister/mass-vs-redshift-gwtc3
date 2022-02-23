# mass-vs-redshift-gwtc3

This repo contains some tools and scripts for exploring the possibility of mass evolution with redshift in GWTC-3.
To start, make sure [conda](https://docs.conda.io/en/latest/miniconda.html) is installed, then create a python environment via

```bash
conda env create -f environment.yml
```

Activate your new environment with

```bash
conda activate mass-vs-redshift-gwtc3
```

This will guarantee that you have all the python libraries and command line tools needed.
Our runs can subsequently be reproduced by moving into the `run/` directory and executing e.g.

```bash
python run_powerLaw_peak_inVsOutPeak.py
```

The results from this run can be inspected within the `run/inspect_plPeak_wRate_inVsOutPeak.ipynb` jupyter notebook.
