import numpy as np 
import os
import sys
import pdb
import gzip



def create_mapping_from_rs_id_to_z_score(input_sumstats_file):
	f = open(input_sumstats_file)
	head_count = 0
	dicti = {}
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		rs_id = data[0]
		short_variant_id = data[1] + '_' + data[2]
		beta = float(data[10])
		se = float(data[11])
		z_score = beta/se
		if se != 0.0:
			dicti[short_variant_id] = z_score
	f.close()
	return dicti



trait_name = sys.argv[1]
samp_size = sys.argv[2]
preprocessed_data_for_sldsc_dir = sys.argv[3]
ukbb_sumstats_hg38_dir = sys.argv[4]

input_sumstats_file = ukbb_sumstats_hg38_dir + trait_name + '_hg38_liftover.bgen.stats'


rs_id_to_z_score = create_mapping_from_rs_id_to_z_score(input_sumstats_file)



output_file = preprocessed_data_for_sldsc_dir + trait_name + '_all_chrom.sumstats'
t = open(output_file,'w')
t.write('SNP\tA1\tA2\tN\tCHISQ\tZ\n')
for chrom_num in range(1,23):
	print(chrom_num)
	file_name = preprocessed_data_for_sldsc_dir + '1000G.EUR.hg38.' + str(chrom_num) + '.frq'
	f = open(file_name)
	g = gzip.open(preprocessed_data_for_sldsc_dir + 'baselineLD.' + str(chrom_num)+ '.annot.gz')
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		line2 =g.next().rstrip()
		data2 = line2.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		rs_id = data[1]
		if data2[2] != rs_id:
			print('assumption erorro')
			pdb.set_trace()
		short_variant_id = data2[0] + '_' + data2[1]
		zscore = rs_id_to_z_score[short_variant_id]

		t.write(rs_id + '\t' + data[2] + '\t' + data[3] + '\t' + samp_size + '\t' + str(np.square(zscore)) + '\t' + str(zscore) + '\n')
	f.close()
	g.close()


t.close()

output_file = preprocessed_data_for_sldsc_dir + trait_name + '_even_chrom.sumstats'
t = open(output_file,'w')
t.write('SNP\tA1\tA2\tN\tCHISQ\tZ\n')
for chrom_num in range(1,23):
	if np.mod(chrom_num, 2) != 0:
		continue
	print(chrom_num)
	file_name = preprocessed_data_for_sldsc_dir + '1000G.EUR.hg38.' + str(chrom_num) + '.frq'
	f = open(file_name)
	g = gzip.open(preprocessed_data_for_sldsc_dir + 'baselineLD.' + str(chrom_num)+ '.annot.gz')
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		line2 =g.next().rstrip()
		data2 = line2.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		rs_id = data[1]
		if data2[2] != rs_id:
			print('assumption erorro')
			pdb.set_trace()
		short_variant_id = data2[0] + '_' + data2[1]
		zscore = rs_id_to_z_score[short_variant_id]

		t.write(rs_id + '\t' + data[2] + '\t' + data[3] + '\t' + samp_size + '\t' + str(np.square(zscore)) + '\t' + str(zscore) + '\n')
	f.close()
	g.close()


t.close()

output_file = preprocessed_data_for_sldsc_dir + trait_name + '_odd_chrom.sumstats'
t = open(output_file,'w')
t.write('SNP\tA1\tA2\tN\tCHISQ\tZ\n')
for chrom_num in range(1,23):
	if np.mod(chrom_num, 2) == 0:
		continue
	print(chrom_num)
	file_name = preprocessed_data_for_sldsc_dir + '1000G.EUR.hg38.' + str(chrom_num) + '.frq'
	f = open(file_name)
	g = gzip.open(preprocessed_data_for_sldsc_dir + 'baselineLD.' + str(chrom_num)+ '.annot.gz')
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		line2 =g.next().rstrip()
		data2 = line2.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		rs_id = data[1]
		if data2[2] != rs_id:
			print('assumption erorro')
			pdb.set_trace()
		short_variant_id = data2[0] + '_' + data2[1]
		zscore = rs_id_to_z_score[short_variant_id]

		t.write(rs_id + '\t' + data[2] + '\t' + data[3] + '\t' + samp_size + '\t' + str(np.square(zscore)) + '\t' + str(zscore) + '\n')
	f.close()
	g.close()


t.close()
