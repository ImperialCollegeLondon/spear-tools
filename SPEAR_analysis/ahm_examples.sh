# setup soft links to make paths easier
ln -s /Volumes/SPEAR_SSD4/ahm_tests_20220728 my_results
ln -s /Volumes/SPEAR_SSD4/SPEAR_S1S2 spear_data


# single example
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_passthrough_$(date '+%Y%m%d_%H%M%S') -m passthrough -l D2 2 M00

# single session
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_passthrough_$(date '+%Y%m%d_%H%M%S') -m passthrough -l D2 2

# whole dataset doesn't work just now


# all baseline
for dataset in D1 D2 D3 D4; do
  for session in 1 2; do
    python spear_enhance.py 'spear_data/Main/Train' my_results/003_testing_baseline_20220801_171800 -m baseline -l $dataset $session
  done
done


# metrics
python spear_evaluate.py 'spear_data/Main/Train' my_results/003_testing_baseline_20220801_171800 segments.csv my_results/004_metrics/baseline.csv


# new examples - 30 Aug 2022

# setup soft links to make paths easier
ln -s /Volumes/SPEAR_SSD4/ahm_tests_20220830 my_results
ln -s /Volumes/SPEAR_SSD4/v2/Volumes/SPEAR_SSD3/SPEAR_v2 spear_data

# all in one go
python spear_enhance.py 'spear_data/Main/Dev' my_results/001_testing_passthrough_20220839_113800 -m passthrough
python spear_enhance.py 'spear_data/Main/Dev' my_results/002_testing_baseline_20220839_124200 -m baseline

# in separate tabs - parallel computations
conda activate spear-tools2
python spear_enhance.py 'spear_data/Main/Dev' my_results/002_testing_baseline_20220839_124200 -m baseline -l D2
python spear_enhance.py 'spear_data/Main/Dev' my_results/002_testing_baseline_20220839_124200 -m baseline -l D3
python spear_enhance.py 'spear_data/Main/Dev' my_results/002_testing_baseline_20220839_124200 -m baseline -l D4
