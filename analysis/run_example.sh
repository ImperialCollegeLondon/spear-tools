# Setup soft links to make paths easier
ln -sf <outputs-folder-full-path> my_results  
ln -sf <dataset-folder-full-path> spear_data  


# Define variables
SET='Dev'
DATASET=2
SESSION='10'
MINUTE=''
METRICS='SDR ISR'
PROCESSING='baseline'
REFERENCE='passthrough'


# Path
input_root_ref="spear_data/Main/$SET"
input_root_proc="spear_data/Main/$SET"

audio_dir_ref="my_results/audio_${REFERENCE}_$(date '+%Y%m%d')"
audio_dir_proc="my_results/audio_${PROCESSING}_$(date '+%Y%m%d')"

segments_csv="segments_$SET.csv"

metrics_csv_ref="my_results/metrics/${REFERENCE}_${SET}_D${DATASET}_S${SESSION}_M${MINUTE}.csv"
metrics_csv_proc="my_results/metrics/${PROCESSING}_${SET}_D${DATASET}_S${SESSION}_M${MINUTE}.csv"

plots_dir="my_results/metrics/${PROCESSING}_${SET}_D${DATASET}_S${SESSION}_M${MINUTE}"


# Run passtrough and enhance
python spear_enhance.py $input_root_ref  $audio_dir_ref  --method_name $REFERENCE --list_cases D$DATASET $SESSION $MINUTE
python spear_enhance.py $input_root_proc $audio_dir_proc --method_name $PROCESSING --list_cases D$DATASET $SESSION $MINUTE

# Compute metrics
python spear_evaluate.py $input_root_ref  $audio_dir_ref  $segments_csv $metrics_csv_ref  --list_cases D$DATASET $SESSION $MINUTE --metrics=$METRICS
python spear_evaluate.py $input_root_proc $audio_dir_proc $segments_csv $metrics_csv_proc --list_cases D$DATASET $SESSION $MINUTE --metrics=$METRICS

# Visualise/plot metrics
python spear_visualise.py $plots_dir $metrics_csv_ref $metrics_csv_proc --reference_name $REFERENCE --method_name $PROCESSING
