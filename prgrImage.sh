#!/bin/bash

# Send an email when important events happen
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=a.gassilloud@student.rug.nl

# Run for at most 30 minutes
#SBATCH --time=00:30:00

# Run on v100, since they have the shortest queue times and to ensure 
# consistency between runs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

# Clean environment
module purge

# Load everything we need for TensorFlow (loads python, tensorflow and a lot more) and scikit
module load TensorFlow/2.6.0-foss-2021a scikit-learn/0.24.2-foss-2021a matplotlib/3.4.2-foss-2021a 

# Run the python script, outputting to a predefined output directory and passing any arguments passed to the bash file
python ~/deep_learning_course/project_2/main.py --log_dir ~/deep_learning_course/project_2/output/ 