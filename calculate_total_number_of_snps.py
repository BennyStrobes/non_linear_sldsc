import numpy as np 
import os
import sys
import pdb








preprocessed_data_for_non_linear_sldsc_dir = sys.argv[1]
trait_name = sys.argv[2]

snp_count = 0
for chrom_num in range(1,23):
	print(chrom_num)
	input_file = preprocessed_data_for_non_linear_sldsc_dir + trait_name + '_genome_wide_susie_windows_and_non_linear_sldsc_processed_data_chrom_' + str(chrom_num) + '.txt'
	head_count = 0
	snp_names = {}

	f = open(input_file)
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		window_name = data[0]
		variant_ids = np.load(data[1])
		for variant_id in variant_ids:
			snp_names[variant_id] = 1
	f.close()

	snp_count = snp_count + len(snp_names)


print(snp_count)