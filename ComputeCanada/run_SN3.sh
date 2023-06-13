#! /bin/bash
#SBATCH --account=def-lplevass                                                             
#SBATCH --time=40:00:00                                                                    
#SBATCH --job-name=NGC4449_SN3                                                      
#SBATCH --output=%x.out                                                                    
#SBATCH --mem-per-cpu=4Gb                                                               
#SBATCH --cpus-per-task=40              
#SBATCH --nodes=1
module load scipy-stack
source luci/bin/activate
python run_SN3.py
