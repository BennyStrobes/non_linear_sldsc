import numpy as np 
import os
import sys
import pdb
import scipy.sparse
import time


def extract_in_sample_variants(ukbb_in_sample_ld_dir, chrom_num):
	for file_name in os.listdir(ukbb_in_sample_ld_dir):
		if file_name.startswith('ukb_imp_v3.c' + chrom_num + '_') == False:
			continue
		full_file_name = ukbb_in_sample_ld_dir + file_name
		pdb.set_trace()

def read_ld(fpath):
	"""
	Read LD files.
		- `_fullld.npy` : full_ld matrix, np.array(dtype=np.float32)
		- `_ld.npz` : ld matrix with SNPs in 10MB window, sp.sparse.csc_matrix(dtype=np.float32)
	Parameters
	----------
	fpath: str
		LD file path.
	Returns
	-------
	mat_ld : np.array(dtype=np.float32) or sp.sparse.csc_matrix(dtype=np.float32)
		LD matrix of dimension (n_ref_snp, n_snp)
	dic_range : dict
		
		- dic_range['chr'] : chromosome
		- dic_range['start'] : start position
		- dic_range['end'] : end position
		- dic_range['chr_ref'] : reference chromosome list (List)      
	"""
	
	# Check fpath
	err_msg = "fpath should end with one of ['_fullld.npy', '_ld.npz'] : %s" % fpath
	assert fpath.endswith("_fullld.npy") | fpath.endswith("_ld.npz"), err_msg
	
	if fpath.endswith("_fullld.npy"):
		mat_ld = np.load(fpath)
		temp_str = [x for x in fpath.split('.') if x.endswith('_fullld')][0]
		dic_range = parse_snp_range(temp_str)
		
	if fpath.endswith("_ld.npz"):
		mat_ld = scipy.sparse.load_npz(fpath)
		temp_str = [x for x in fpath.split('.') if x.endswith('_ld')][0]
		dic_range = parse_snp_range(temp_str)

	return mat_ld,dic_range


def parse_snp_range(snp_range):
	"""Get range of SNPs to analyze. 
	
	Parameters
	----------
	snp_range: str
		Example: 'c1_s0_e2000_r1'
	Returns
	-------
	dic_range : dict
		
		- dic_range['chr'] : chromosome
		- dic_range['start'] : start position
		- dic_range['end'] : end position
		- dic_range['chr_ref'] : reference chromosome list (List)
	"""

	dic_range = {x: None for x in ["chr", "start", "end", "chr_ref"]}

	for x in snp_range.split("_"):

		if x[0] == "c":
			dic_range["chr"] = int(x.replace("c", "").strip())

		if x[0] == "s":
			dic_range["start"] = int(x.replace("s", "").strip())

		if x[0] == "e":
			dic_range["end"] = int(x.replace("e", "").strip())

		if x[0] == "r":
			temp_str = x.replace("r", "").strip()
			if temp_str == "all":
				dic_range["chr_ref"] = list(np.arange(1, 23))
			else:
				dic_range["chr_ref"] = [int(x) for x in temp_str.split(",")]

	return dic_range

def create_mapping_from_rsid_to_in_sample_variant_index(chrom_pvar_file):
	f = open(chrom_pvar_file)
	dicti = {}
	rs_id_to_alleles = {}
	head_count = 0
	indexer = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		rsid = data[2]
		alleles = data[4] + '_' + data[3]
		if rsid in rs_id_to_alleles:
			print('assumption eroror')
			pdb.set_trace()
		rs_id_to_alleles[rsid] = alleles
		if rsid in dicti:
			print('assumption eroror')
			pdb.set_trace()
		dicti[rsid] = indexer
		indexer = indexer + 1
	f.close()
	return dicti, rs_id_to_alleles
def extract_overlapping_variants(window_rsids, rs_id_to_in_sample_variant):
	valid_window_indices = []
	in_sample_variant_indices = []

	for ii, window_rsid in enumerate(window_rsids):
		if window_rsid in rs_id_to_in_sample_variant:
			valid_window_indices.append(ii)
			in_sample_variant_indices.append(rs_id_to_in_sample_variant[window_rsid])

	if len(valid_window_indices) != len(in_sample_variant_indices):
		print('assumptino eroror')
		pdb.set_trace()

	if len(in_sample_variant_indices) != len(np.unique(in_sample_variant_indices)):
		print('assumptione rororo')
		pdb.set_trace()

	return np.asarray(valid_window_indices), np.asarray(in_sample_variant_indices)

def indices_dont_lie_in_file(sample_ld_variant_indices, file_index_start, file_index_end):
	booler = True
	for indexer in sample_ld_variant_indices:
		if indexer >= file_index_start and indexer < file_index_end:
			booler = False
	return booler

def extract_ld_mat_from_in_sample_ld(sample_ld_variant_indices, ukbb_in_sample_ld_dir, chrom_num):
	min_index = np.min(sample_ld_variant_indices)
	max_index = np.max(sample_ld_variant_indices)

	num_var = len(sample_ld_variant_indices)

	ld_mat = np.zeros((num_var, num_var)) + -2000.0

	for file_name in os.listdir(ukbb_in_sample_ld_dir):
		if file_name.startswith('ukb_imp_v3_chimp.c' + chrom_num + '_') == False:
			continue
		if file_name.endswith('.compute_ld.sbatch.log'):
			continue
		full_file_name = ukbb_in_sample_ld_dir + file_name

		file_info = file_name.split('_')


		file_index_start = int(file_info[4].split('s')[1])
		file_index_end = int(file_info[5].split('e')[1])

		if indices_dont_lie_in_file(sample_ld_variant_indices, file_index_start, file_index_end):
			continue

		file_indices = []
		col_names = []
		for ii, sample_index in enumerate(sample_ld_variant_indices):
			if sample_index >= file_index_start and sample_index < file_index_end:
				file_indices.append(ii)
				col_names.append(sample_index - file_index_start)
		file_indices = np.asarray(file_indices)
		col_names = np.asarray(col_names)

		sparse_mat, sparse_mat_info = read_ld(full_file_name)
		ld_mat[:, file_indices] = (sparse_mat[sample_ld_variant_indices,:][:,col_names]).toarray()

	if np.sum(ld_mat == -2000.0) > 0:
		print('assumption eroror')
		pdb.set_trace()
	return ld_mat

chrom_num = sys.argv[1]
ukbb_preprocessed_for_genome_wide_susie_dir = sys.argv[2]
ukbb_in_sample_ld_dir = sys.argv[3]
ukbb_in_sample_genotype_dir = sys.argv[4]




# RS_ID to in_sample variant INDEX
chrom_pvar_file = ukbb_in_sample_genotype_dir + 'ukb_imp_chr' + chrom_num + '_v3_chimp.pvar'
rs_id_to_in_sample_variant, rs_id_to_in_sample_alleles = create_mapping_from_rsid_to_in_sample_variant_index(chrom_pvar_file)


input_window_file = ukbb_preprocessed_for_genome_wide_susie_dir + 'genome_wide_susie_windows_and_processed_data_chrom_' + chrom_num + '.txt'
output_window_file2 = ukbb_preprocessed_for_genome_wide_susie_dir + 'genome_wide_susie_windows_and_processed_data_big_windows_chrom_' + chrom_num + '.txt'
output_window_file = ukbb_preprocessed_for_genome_wide_susie_dir + 'genome_wide_susie_windows_and_processed_data_in_sample_ld_chrom_' + chrom_num + '.txt'
# Output window file containing in sample ld
t = open(output_window_file,'w')
t2 = open(output_window_file2,'w')

# Loop through windows:
f = open(input_window_file)
head_count = 0
for line in f:
	line = line.rstrip()
	data = line.split('\t')
	if head_count == 0:
		# Header
		head_count = head_count + 1
		data[9] = 'genotype_ld_file'
		t.write('\t'.join(data) + '\n')
		t2.write(line + '\n')
		continue

	start_time = time.time()
	# Standard line
	#extract relevent fields
	line_chrom_num = data[0]
	window_start = data[1]
	window_end = data[2]
	start_bool = data[3]
	end_bool = data[4]
	beta_file = data[5]
	beta_se_file = data[6]
	variant_file = data[7]
	tissue_file = data[8]
	sample_size_file = data[10]
	cm_file = data[11]
	rsid_file = data[12]

	window_name = line_chrom_num + ':' + window_start + ':' + window_end

	# Previous window rs_ids
	window_rsids = np.loadtxt(rsid_file, dtype=str)


	# Extract valid window indices, as well as sample ld variant indices corresponding to those
	valid_window_indices, sample_ld_variant_indices = extract_overlapping_variants(window_rsids, rs_id_to_in_sample_variant)


	if len(valid_window_indices) < 5000:
		print('************')
		print(len(window_rsids))
		print(len(valid_window_indices))
		continue
	print('################')
	print(len(window_rsids))
	print(len(valid_window_indices))
	t2.write(line + '\n')

	# Now filter data files per new valid_window_indices
	# beta
	beta = np.loadtxt(beta_file)
	beta = beta[:, valid_window_indices]
	# beta se
	beta_se = np.loadtxt(beta_se_file)
	beta_se = beta_se[:, valid_window_indices]
	# variant
	variants = np.loadtxt(variant_file, dtype=str)
	variants = variants[valid_window_indices]
	# CM
	cm = np.loadtxt(cm_file)
	cm = cm[valid_window_indices]
	# rsids
	rsids = window_rsids[valid_window_indices]

	# NOW GET LD matrix
	ld_mat = extract_ld_mat_from_in_sample_ld(sample_ld_variant_indices, ukbb_in_sample_ld_dir, chrom_num)


	# Save to output file
	# Beta file
	new_beta_file = ukbb_preprocessed_for_genome_wide_susie_dir + window_name + '_in_sample_ld_beta.txt'
	np.savetxt(new_beta_file, beta, fmt="%s", delimiter='\t')
	# stderr file
	new_stderr_file = ukbb_preprocessed_for_genome_wide_susie_dir + window_name + '_in_sample_ld_beta_std_err.txt'
	np.savetxt(new_stderr_file, beta_se, fmt="%s", delimiter='\t')
	# Variant file
	new_variant_file = ukbb_preprocessed_for_genome_wide_susie_dir + window_name + '_in_sample_ld_variant_ids.txt'
	np.savetxt(new_variant_file, variants, fmt="%s", delimiter='\t')
	# CM file
	new_cm_file = ukbb_preprocessed_for_genome_wide_susie_dir + window_name + '_in_sample_ld_variant_cm.txt'
	np.savetxt(new_cm_file, cm, fmt="%s", delimiter='\t')
	# rsid file
	new_rsid_file = ukbb_preprocessed_for_genome_wide_susie_dir + window_name + '_in_sample_ld_variant_rsid.txt'
	np.savetxt(new_rsid_file, rsids, fmt="%s", delimiter='\t')	
	# LD file
	ld_file = ukbb_preprocessed_for_genome_wide_susie_dir + window_name + '_in_sample_ld_ld.npy'
	np.save(ld_file, ld_mat)	

	t.write('\t'.join(data[:5]) + '\t' + new_beta_file + '\t' + new_stderr_file + '\t' + new_variant_file + '\t' + tissue_file + '\t' + ld_file + '\t' + sample_size_file + '\t' + new_cm_file + '\t' + new_rsid_file + '\n')

	end_time = time.time()
	print(end_time-start_time)

f.close()

t.close()
t2.close()




