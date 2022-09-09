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
python spear_enhance.py 'spear_data/Main/Dev' my_results/002_testing_baseline_20220839_124200 -m baseline -l D1
python spear_enhance.py 'spear_data/Main/Dev' my_results/002_testing_baseline_20220839_124200 -m baseline -l D2
python spear_enhance.py 'spear_data/Main/Dev' my_results/002_testing_baseline_20220839_124200 -m baseline -l D3
python spear_enhance.py 'spear_data/Main/Dev' my_results/002_testing_baseline_20220839_124200 -m baseline -l D4


python spear_evaluate.py 'spear_data/Main/Dev' my_results/001_testing_passthrough_20220839_113800 segments_Dev_D1_S10.csv my_results/003_metrics/baseline_D1_S10.csv




# new examples 2 Sep 2022
# make softlinks so that we can use short, relative paths in subsequent commands
ln -s /Volumes/SPEAR_SSD4/ahm_tests_20220902 my_results
ln -s /Volumes/SPEAR_SSD4/v20220901 spear_data

conda activate spear-tools2

for dataset in D1 D2 D3 D4; do
  python spear_enhance.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 -m passthrough -l $dataset 1
done

# run commands in separate tabs to reduce elapsed time
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 -m baseline -l D1 1
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 -m baseline -l D2 1
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 -m baseline -l D3 1
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 -m baseline -l D4 1

# run commands in separate tabs to reduce elapsed time
python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D1_S1.csv my_results/003_metrics/passthrough_D1_S1.csv
python spear_evaluate.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 segments_Train_D1_S1.csv my_results/003_metrics/baseline_D1_S1.csv

python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D2_S1.csv my_results/003_metrics/passthrough_D2_S1.csv
python spear_evaluate.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 segments_Train_D2_S1.csv my_results/003_metrics/baseline_D2_S1.csv


python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D3_S1.csv my_results/003_metrics/passthrough_D3_S1.csv
python spear_evaluate.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 segments_Train_D3_S1.csv my_results/003_metrics/baseline_D3_S1.csv

python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D4_S1.csv my_results/003_metrics/passthrough_D4_S1.csv
python spear_evaluate.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 segments_Train_D4_S1.csv my_results/003_metrics/baseline_D4_S1.csv


for dataset in D1 D2 D3 D4; do
  python spear_enhance.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 -m passthrough -l $dataset 2
done

python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 -m baseline -l D1 2
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 -m baseline -l D2 2
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 -m baseline -l D3 2
python spear_enhance.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 -m baseline -l D4 2


python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D1_S2.csv my_results/003_metrics/passthrough_D1_S2.csv
python spear_evaluate.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 segments_Train_D1_S2.csv my_results/003_metrics/baseline_D1_S2.csv

python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D2_S2.csv my_results/003_metrics/passthrough_D2_S2.csv
python spear_evaluate.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 segments_Train_D2_S2.csv my_results/003_metrics/baseline_D2_S2.csv

python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D3_S2.csv my_results/003_metrics/passthrough_D3_S2.csv
python spear_evaluate.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 segments_Train_D3_S2.csv my_results/003_metrics/baseline_D3_S2.csv

python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D4_S2.csv my_results/003_metrics/passthrough_D4_S2.csv
python spear_evaluate.py 'spear_data/Main/Train' my_results/002_testing_baseline_20220902_060500 segments_Train_D4_S2.csv my_results/003_metrics/baseline_D4_S2.csv


  
python spear_evaluate.py 'spear_data/Main/Train' my_results/001_testing_passthrough_20220902_060000 segments_Train_D4_S2_debug2.csv my_results/003_metrics/passthrough_D4_S2_debug2_again.csv --metrics="SDR ISR SAR SI-SDR PESQ PESQ-NB"
