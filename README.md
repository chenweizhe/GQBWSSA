# GQBWSSA IDS Feature Selection

Paper: Large-scale IoT attack detection scheme based on LightGBM and feature selection using an improved salp swarm algorithm

The original project was implemented in MATLAB and has been ported to Python for easier reproduction. In the fitness function, the random forest is trained and used for prediction on the GPU with CuPy/CuML, which is faster than the MATLAB training. The comparison algorithms are GWO, WOA, DBO, and HHO.

---

## 1. Quick start

Run the commands below from the `python` directory:

```powershell
# Run GQBWSSA only
python main.py --algos GQBWSSA

# Choose algorithms, the number of repeats, and a random seed
python main.py --seed 46 --algos GWO WOA GQBWSSA
```

---

## 2. Fitness function

The fitness function is the same as in `Fitness_RF.m`:

```
fit = 0.99 × (1 − ACC) + 0.01 × feature selection ratio
```

---

## 3. Other differences from the MATLAB version

- **Random numbers**: Some element-wise random-number loops now generate a whole row at once. The distribution is identical, but the random sequence is not.
- **HHO**: The original code evaluated the fitness of the same individual `X(i,:)` twice within the same branch. The Python version evaluates it once and reuses the result, so the random forest is trained one fewer time.
- **Rounding**: `round` in GQBWSSA follows MATLAB's "round half away from zero" rule, rather than NumPy's default banker's rounding.
- Numerical precision may differ between Python and MATLAB.

---
