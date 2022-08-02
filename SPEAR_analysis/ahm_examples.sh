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
