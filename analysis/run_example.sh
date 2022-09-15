# Uncomment and edit the following lines to create symbolic links so you can choose
# where to keep your data
# ln -s <dataset-folder-full-path> spear_data  
# ln -s <outputs-folder-full-path> my_results  


# Define variables
SET='Dev'
DATASET=2
SESSION='10'
MINUTE='00'
METRICS='SDR ISR'
PROCESSING='baseline'
REFERENCE='passthrough'


# Output paths
# These can be whatever you want them to be
audio_dir_ref="my_results/audio_${REFERENCE}_$(date '+%Y%m%d')"
audio_dir_proc="my_results/audio_${PROCESSING}_$(date '+%Y%m%d')"
metrics_dir="my_results/metrics"
plots_dir="my_results/plots/${PROCESSING}_${SET}_D${DATASET}_S${SESSION}_M${MINUTE}"

mkdir $audio_dir_ref
mkdir $audio_dir_proc
mkdir $metrics_dir
mkdir -p $plots_dir


# Input paths relative to spear_data should not be changed
input_root_ref="spear_data/Main/$SET"
input_root_proc="spear_data/Main/$SET"
segments_csv="segments_$SET.csv"

# Derived paths
metrics_csv_ref="${metrics_dir}/${REFERENCE}_${SET}_D${DATASET}_S${SESSION}_M${MINUTE}.csv"
metrics_csv_proc="${metrics_dir}/${PROCESSING}_${SET}_D${DATASET}_S${SESSION}_M${MINUTE}.csv"



# Run passtrough and enhance
python spear_enhance.py $input_root_ref  $audio_dir_ref  --method_name $REFERENCE --list_cases D$DATASET $SESSION $MINUTE
python spear_enhance.py $input_root_proc $audio_dir_proc --method_name $PROCESSING --list_cases D$DATASET $SESSION $MINUTE

# Compute metrics
python spear_evaluate.py $input_root_ref  $audio_dir_ref  $segments_csv $metrics_csv_ref  --list_cases D$DATASET $SESSION $MINUTE --metrics=$METRICS
python spear_evaluate.py $input_root_proc $audio_dir_proc $segments_csv $metrics_csv_proc --list_cases D$DATASET $SESSION $MINUTE --metrics=$METRICS

# Visualise/plot metrics
python spear_visualise.py $plots_dir $metrics_csv_ref $metrics_csv_proc --reference_name $REFERENCE --method_name $PROCESSING
