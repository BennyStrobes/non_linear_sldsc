import numpy as np 
import os
import sys
import pdb




def extract_sample_size_and_h2_from_log_file(bolt_lmm_log_file):
	g = open(bolt_lmm_log_file)
	samp_size = -1
	h2 = -3000
	for line in g:
		line = line.rstrip()
		# Heritability line
		if line.startswith('Estimated (pseudo-)heritability:'):
			h2 = float(line.split(' = ')[1])
		if line.startswith('Number of indivs with no missing phenotype(s) to use'):
			samp_size = float(line.split('use: ')[1])
	g.close()
	if samp_size == -1 or h2 == -3000:
		print('assumption in extracting h2 or samp size')
		pdb.set_trace()

	return int(samp_size), h2


sumstats_dir = sys.argv[1]
sumstat_summary_file = sys.argv[2]

# get study names
study_names = []
for file_name in os.listdir(sumstats_dir):
	if file_name.endswith('.bgen.stats.gz') == False:
		continue
	study_name = file_name.split('.bgen.stats.gz')[0].split('v3.')[1]
	if study_name == '460K.CAD':
		continue
	if study_name.endswith('interim'):
		continue
	study_names.append(study_name)
study_names = np.unique(study_names)

t = open(sumstat_summary_file,'w')

for trait_name in study_names:
	# Log file for this trait from bolt-lmm
	trait_file = sumstats_dir + 'bolt_337K_unrelStringentBrit_MAF0.001_v3.' + trait_name + '.bgen.stats.gz'
	# Extact sample size and h2 from log file
	bolt_lmm_log_file = sumstats_dir + 'bolt_337K_unrelStringentBrit_MAF0.001_v3.' + trait_name + '.sbatch.log'

	# Extact sample size and h2 from log file
	sample_size, h2 = extract_sample_size_and_h2_from_log_file(bolt_lmm_log_file)
	
	# Min heritability threshold
	if h2 > .1:
		# print to output
		t.write(trait_name + '\t' + trait_file + '\t' + str(sample_size) + '\t' + str(h2) + '\n')

t.close()

'''
input_trait_file = output_dir + 'ukbb_hg38_sumstat_files.txt'

output_trait_file = output_dir + 'ukbb_hg38_sumstat_files_with_samp_size_and_h2.txt'

f = open(input_trait_file)
t = open(output_trait_file,'w')

head_count = 0

for line in f:
	line = line.rstrip()
	data = line.split('\t')
	if head_count == 0:
		head_count = head_count + 1
		t.write(data[0] + '\t' + data[1] + '\t' + 'sample_size\tbolt_lmm_h2\n')
		continue
	# Extract relevent fields from line
	trait_name = data[0]
	trait_file = data[1]
	trait_file = output_dir + trait_file.split('/')[-1]

	# Log file for this trait from bolt-lmm
	bolt_lmm_log_file = input_dir + 'bolt_337K_unrelStringentBrit_MAF0.001_v3.' + trait_name + '.sbatch.log'

	# Extact sample size and h2 from log file
	sample_size, h2 = extract_sample_size_and_h2_from_log_file(bolt_lmm_log_file)

	# Min heritability threshold
	if h2 > .1:
		# print to output
		t.write(trait_name + '\t' + trait_file + '\t' + str(sample_size) + '\t' + str(h2) + '\n')

f.close()
t.close()
'''