import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import numpy as np 
import pandas
import os
import pdb
import gzip



def extract_variants(non_linear_window_file):
	f = open(non_linear_window_file)
	head_count = 0
	dicti = {}
	dicti_short = {}
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		variant_id_file = data[1]
		variant_ids = np.load(variant_id_file)
		for variant_id in variant_ids:
			dicti[variant_id] = 1
			dicti_short['_'.join(variant_id.split('_')[:2])] = 1
	f.close()
	if len(dicti) != len(dicti_short):
		print('fundamental assumptino errror')
		pdb.set_trace()
	return dicti, dicti_short


def filter_annotation_file(old_annotation_file, new_annotation_file, short_variant_list):
	f = gzip.open(old_annotation_file)
	t = gzip.open(new_annotation_file,'w')

	mapping_from_short_variant_id_to_rsid = {}
	rs_id_list = {}

	head_count = 0
	used_var = {}
	for line in f:
		line1 = line.decode("utf-8").rstrip()
		data = line1.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			t.write(line)
			continue
		short_variant_id = 'chr' + data[0] + '_' + data[1]
		if short_variant_id not in short_variant_list:
			continue
		# A few repeats
		if short_variant_id in used_var:
			print('repeat')
			continue
		t.write(line)
		used_var[short_variant_id] = 1
		mapping_from_short_variant_id_to_rsid[short_variant_id] = data[2]
		rs_id_list[data[2]] = 1
	f.close()
	t.close()
	if len(used_var) != len(short_variant_list):
		print('assumption eroror')
		pdb.set_trace()
	if len(mapping_from_short_variant_id_to_rsid) != len(rs_id_list):
		print('assumption erororor')
		pdb.set_trace()
	return mapping_from_short_variant_id_to_rsid, rs_id_list

def generate_weights_file(non_linear_window_file, mapping_from_short_variant_id_to_rsid, new_weight_file):
	# Open output file handle
	t = gzip.open(new_weight_file,'wt')
	t.write('CHR\tSNP\tBP\tL2\n')
	
	# Loop through windows
	f = open(non_linear_window_file)
	head_count = 0
	counter = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		counter = counter +1

		# Load in relevent info
		variant_ids = np.load(data[1])
		regression_indices = np.load(data[11])
		squared_ld = np.load(data[12])
		squared_ld_reg_only = squared_ld[regression_indices, :][:, regression_indices]
		for index, regression_index in enumerate(regression_indices):
			# Extract variant info for variant corresponding to regression index
			variant_id = variant_ids[regression_index]
			variant_info = variant_id.split('_')
			short_variant_id = variant_info[0] + '_' + variant_info[1]
			rs_id = mapping_from_short_variant_id_to_rsid[short_variant_id]
			var_pos = variant_info[1]
			var_chrom_num = variant_info[0].split('hr')[1]
			weight = np.sum(squared_ld_reg_only[index,:])
			t.write(var_chrom_num + '\t' + rs_id + '\t' + var_pos + '\t' + str(weight) + '\n')
	f.close()
	t.close()



def generate_annotation_weighted_ld_score_file(non_linear_window_file, mapping_from_short_variant_id_to_rsid, old_ld_score_file, new_ld_score_file):
	# First get header from old file
	f = gzip.open(old_ld_score_file)
	head_count = 0
	for line in f:
		line = line.decode("utf-8").rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			header = data
			continue
		break
	f.close()

	# Open output file handle
	t = gzip.open(new_ld_score_file,'wt')
	t.write('\t'.join(header) + '\n')

	# Loop through windows
	f = open(non_linear_window_file)
	head_count = 0
	counter = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		counter = counter +1

		# Load in relevent info
		variant_ids = np.load(data[1])
		anno = np.load(data[6])
		regression_indices = np.load(data[11])
		squared_ld = np.load(data[12])
		for regression_index in regression_indices:
			# Extract variant info for variant corresponding to regression index
			variant_id = variant_ids[regression_index]
			variant_info = variant_id.split('_')
			short_variant_id = variant_info[0] + '_' + variant_info[1]
			rs_id = mapping_from_short_variant_id_to_rsid[short_variant_id]
			var_pos = variant_info[1]
			var_chrom_num = variant_info[0].split('hr')[1]

			# Extract annotation weighted ld scores for this variant
			annotation_weighted_ld_scores = np.dot(squared_ld[regression_index,:],anno)

			# Print to output file
			t.write(var_chrom_num + '\t' + rs_id + '\t' + var_pos + '\t' + '\t'.join(annotation_weighted_ld_scores.astype(str)) + '\n')
	f.close()
	t.close()

def filter_frequency_file(old_frq_file, new_frq_file, rs_id_list):
	f = open(old_frq_file)
	t = open(new_frq_file,'w')
	rs_id_to_frq = {}
	frqs = []
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split()
		if head_count == 0:
			head_count = head_count + 1
			t.write(line + '\n')
			continue
		line_rs_id = data[1]
		if line_rs_id not in rs_id_list:
			continue
		frq = float(data[4])
		t.write(line + '\n')
		rs_id_to_frq[line_rs_id] = frq
		frqs.append(frq)
	f.close()
	t.close()
	return rs_id_to_frq

def generate_M_file(new_annotation_file, rs_id_to_frq, af_threshold, m_output_file):
	f = gzip.open(new_annotation_file)
	head_count = 0
	for line in f:
		line = line.decode("utf-8").rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			num_anno = len(data[4:])
			anno_counts = np.zeros(num_anno)
			continue
		rs_id = data[2]
		line_frq = rs_id_to_frq[rs_id]
		if line_frq < af_threshold:
			continue
		anno_counts = anno_counts + np.asarray(data[4:]).astype(float)
	f.close()

	t = open(m_output_file,'w')
	t.write('\t'.join(anno_counts.astype(str)) + '\n')
	t.close()

def generate_weights_M_file(new_weight_file, rs_id_to_frq, af_threshold, weights_m_output_file):
	f = gzip.open(new_weight_file)
	head_count = 0
	counter = 0
	for line in f:
		line = line.decode("utf-8").rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		rs_id = data[1]
		line_frq = rs_id_to_frq[rs_id]
		if line_frq < af_threshold:
			continue
		counter = counter + 1
	f.close()

	t = open(weights_m_output_file,'w')
	t.write(str(counter) + '\n')
	t.close()



ukbb_preprocessed_for_genome_wide_susie_dir = sys.argv[1]
ldsc_baseline_ld_hg38_annotation_dir = sys.argv[2]
preprocessed_data_for_non_linear_sldsc_dir = sys.argv[3]
chrom_num = sys.argv[4]
preprocessed_data_for_sldsc_dir = sys.argv[5]
ref_1kg_genotype_dir = sys.argv[6]


# Extract variants we have used in non-linear analysis
non_linear_window_file = preprocessed_data_for_non_linear_sldsc_dir + 'blood_WHITE_COUNT_genome_wide_susie_windows_and_non_linear_sldsc_processed_data_chrom_' + chrom_num + '.txt'
variant_list, short_variant_list = extract_variants(non_linear_window_file)


# Filter annotation file to be left with new annotation file
old_annotation_file = ldsc_baseline_ld_hg38_annotation_dir + 'baselineLD.' + chrom_num + '.annot.gz'
new_annotation_file = preprocessed_data_for_sldsc_dir + 'baselineLD.' + chrom_num + '.annot.gz'
mapping_from_short_variant_id_to_rsid, rs_id_list = filter_annotation_file(old_annotation_file, new_annotation_file, short_variant_list)

# Create LD Score file
old_ld_score_file = ldsc_baseline_ld_hg38_annotation_dir + 'baselineLD.' + chrom_num + '.l2.ldscore.gz'
new_ld_score_file = preprocessed_data_for_sldsc_dir + 'baselineLD.' + chrom_num + '.l2.ldscore.gz'
generate_annotation_weighted_ld_score_file(non_linear_window_file, mapping_from_short_variant_id_to_rsid, old_ld_score_file, new_ld_score_file)

# Create weights file
new_weight_file = preprocessed_data_for_sldsc_dir + 'weights.hm3_noMHC.' + chrom_num + '.l2.ldscore.gz'
generate_weights_file(non_linear_window_file, mapping_from_short_variant_id_to_rsid, new_weight_file)

old_frq_file = ref_1kg_genotype_dir + '1000G.EUR.hg38.' + chrom_num + '.frq'
new_frq_file = preprocessed_data_for_sldsc_dir + '1000G.EUR.hg38.' + chrom_num + '.frq'
rs_id_to_frq = filter_frequency_file(old_frq_file, new_frq_file, rs_id_list)

m_output_file = preprocessed_data_for_sldsc_dir + 'baselineLD.' + chrom_num + '.l2.M'
generate_M_file(new_annotation_file, rs_id_to_frq, 0.0, m_output_file)

m_output_file = preprocessed_data_for_sldsc_dir + 'baselineLD.' + chrom_num + '.l2.M_5_50'
generate_M_file(new_annotation_file, rs_id_to_frq, 0.05, m_output_file)


weights_m_output_file = preprocessed_data_for_sldsc_dir + 'weights.hm3_noMHC.' + chrom_num + '.l2.M'
generate_weights_M_file(new_weight_file, rs_id_to_frq, 0.0, weights_m_output_file)

weights_m_output_file = preprocessed_data_for_sldsc_dir + 'weights.hm3_noMHC.' + chrom_num + '.l2.M_5_50'
generate_weights_M_file(new_weight_file, rs_id_to_frq, 0.05, weights_m_output_file)






