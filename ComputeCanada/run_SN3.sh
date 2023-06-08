#! /bin/bash
#SBATCH --account=def-lplevass                                                             
#SBATCH --time=40:00:00                                                                    
#SBATCH --job-name=NGC4449_SN3                                                      
#SBATCH --output=%x.out                                                                    
#SBATCH --mem-per-cpu=16Gb                                                               
#SBATCH --cpus-per-task=1               # maximum cpu per task is 3.5 per gpus 
#SBATCH --tasks=100
#SBATCH --ntasks-per-node=25
module load cuda cudnn 
module load scipy-stack
source luci/bin/activate
python run_SN3.py
