# DeepMMR

## Overview

Python implementation of Deep Maximum Moment Restriction (DeepMMR).

## Steps to run the experiments

1. Install virtual environment
   ```bash
   conda create -n DeepMMR python=3.12
   ```
2. Activate environment
   ```bash
   conda activate DeepMMR
   ```
3. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
4. Run simulation experiments (main.py)
   
   The script includes experiment calls for multiple models. You can uncomment or modify the lines depending on the scenarios and configurations you want to run.

   **Parameters Description**
   - `repeat_times`: Number of times to repeat the experiment
   - `scenario`: Simulation scenario ('S1', 'S2', ..., 'S6')
   - `train_size`: Sample size of the training dataset
   - `test_size`: Sample size of the test dataset
   - `save_path`: Directory to save experiment results
   - `type`: Type of statistics used ('u': U-statistic, 'v': V-statistic)

   **GMM: Generalized Method of Moments**
   ```python
   # run_gmm_experiments(repeat_times, scenario, train_size, test_size, save_path)
   run_gmm_experiments(100, 'S1', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'GMM'))
   ```
   
   **PMMR: Proximal Maximum Moment Restriction**
   ```python
   # run_pmmr_experiments(repeat_times, scenario, train_size, test_size, save_path)
   run_pmmr_experiments(100, 'S1', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'PMMR'))
   ```
   
   **Minmax: Min-Max optimization**
   ```python
   # run_minmax_experiments(repeat_times, scenario, train_size, test_size, save_path)
   run_minmax_experiments(100, 'S1', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'MinMax'))
   ```
   
   **DeepMMR: Deep Maximum Moment Restriction (Ours)**
   ```python
   # run_mmr_experiments(repeat_times, scenario, type, train_size, test_size, save_path)
   run_mmr_experiments(100, 'S1', 'u', 2000, 1000, str(Path.cwd() / 'results' / 'simulation' / 'DeepMMR'))
   ```
5. Run RHC experiments  (main.py)

   The RHC experiment runs on real-world clinical data.

   **Parameters Description**
   - `repeat_times`: Number of times to repeat the experiment
   - `type`: Type of statistics to be used ('u' represents U-statistic, 'v' represents V-statistic)
   - `save_path`: Directory to save experiment results
   
   ```python
   # run_mmr_rhc(repeat_times, type, train_size, test_size, save_path)
   run_mmr_rhc(100, "u", str(Path.cwd() / 'results' / 'rhc')) 
   ```