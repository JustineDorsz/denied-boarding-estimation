# Estimation of fail-to-board probabilities from AFC and AVL data.

## Installation

- Creation of virtual environmnent to avoid library version conflicts:
```bash
python3 -m venv venv
source venv/bin/activate
```
- Dependencies installation: 
```bash
pip install -r requirements.txt
```
and to see itself as a package:
```bash
pip install -e .
```

- If `matplotlib.pyplot.plot()` fails:
```
sudo apt-get install python3-tk
```

## Model parameters.
Set parameters of walking speed and walking distance in `parameters.yml`.


## Launch MLE estimation

In `main.py` file, set:
- `PATHS` according to the localization of AVL and AFC data
- `station_origin`, `station_destination`, `date`, `direction`.

Run file.
