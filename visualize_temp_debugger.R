args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(hash)

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}


make_ldsc_log_like_barplot <- function(df) {
	#df <- df[as.character(df$model) != "sldsc_linear_model",]
    p <- ggplot(data=df, aes(x=model, y=sum_gamma_log_like)) +
    geom_point(stat="identity") +
    figure_theme() +
    theme(legend.position="top") +
    theme(axis.text.x = element_text(angle = 90,hjust=1, vjust=.5, size=12)) +
    labs(y="Log-LSS gamma loss", x="model", fill="") +
    geom_errorbar(aes(ymin=sum_gamma_log_like-(1.96*se_sum_gamma_log_like), ymax=sum_gamma_log_like+(1.96*se_sum_gamma_log_like)), position = position_dodge(), width = .75, size=.2)

    return(p)
}

make_violin_plot_show_expected_per_snp_chi_squared <- function(model_data_arr, annotation_model_types) {
	per_snp_her_arr <- c()
	model_arr <- c()
	print(annotation_model_types)
	for (model_iter in 1:length(annotation_model_types)) {
		model_type <- annotation_model_types[model_iter]
		model_per_snp_heritabilities <- model_data_arr[[model_iter]]$pred_chi_sq

		per_snp_her_arr <- c(per_snp_her_arr, model_per_snp_heritabilities)
		model_arr <- c(model_arr, rep(model_type, length(model_per_snp_heritabilities)))
	}

	df <- data.frame(model_per_snp_heritabilities=per_snp_her_arr, her_model=factor(model_arr, levels=annotation_model_types))

	p <- ggplot(df, aes(x=her_model, y=model_per_snp_heritabilities)) + 
  		geom_violin(trim=FALSE) +
  		figure_theme() +
  		theme(axis.text.x = element_text(angle = 90,hjust=1, vjust=.5, size=10)) +
  		labs(y="Expected per-snp chi-squared", x="heritability model", fill="")
  	return(p)
}

make_violin_plot_show_expected_per_snp_log_chi_squared <- function(model_data_arr, annotation_model_types) {
	per_snp_her_arr <- c()
	model_arr <- c()
	print(annotation_model_types)
	for (model_iter in 1:length(annotation_model_types)) {
		model_type <- annotation_model_types[model_iter]
		model_per_snp_heritabilities <- model_data_arr[[model_iter]]$pred_chi_sq

		per_snp_her_arr <- c(per_snp_her_arr, model_per_snp_heritabilities)
		model_arr <- c(model_arr, rep(model_type, length(model_per_snp_heritabilities)))
	}

	df <- data.frame(model_per_snp_heritabilities=per_snp_her_arr, her_model=factor(model_arr, levels=annotation_model_types))
	print(summary(df))
	p <- ggplot(df, aes(x=her_model, y=model_per_snp_heritabilities)) + 
  		geom_violin(trim=FALSE) +
  		figure_theme() +
  		scale_y_continuous(trans='log') +
  		theme(axis.text.x = element_text(angle = 90,hjust=1, vjust=.5, size=10)) +
  		labs(y="Expected per-snp chi-squared", x="heritability model", fill="")
  	return(p)
}


make_violin_plot_show_expected_per_snp_sldsc_log_like <- function(model_data_arr, annotation_model_types) {
	per_snp_her_arr <- c()
	model_arr <- c()
	print(annotation_model_types)
	for (model_iter in 1:length(annotation_model_types)) {
		model_type <- annotation_model_types[model_iter]
		model_per_snp_heritabilities <- model_data_arr[[model_iter]]$ldsc_gamma_log_like

		per_snp_her_arr <- c(per_snp_her_arr, model_per_snp_heritabilities)
		model_arr <- c(model_arr, rep(model_type, length(model_per_snp_heritabilities)))
	}

	df <- data.frame(model_per_snp_heritabilities=per_snp_her_arr, her_model=factor(model_arr, levels=annotation_model_types))

	p <- ggplot(df, aes(x=her_model, y=model_per_snp_heritabilities)) + 
  		geom_violin(trim=FALSE) +
  		figure_theme() +
  		theme(axis.text.x = element_text(angle = 90,hjust=1, vjust=.5, size=10)) +
  		labs(y="Expected per-snp sldsc gamma log like", x="heritability model", fill="")
  	return(p)
}





make_violin_plot_show_expected_per_snp_heritabilities <- function(model_data_arr, annotation_model_types) {
	per_snp_her_arr <- c()
	model_arr <- c()
	print(annotation_model_types)
	for (model_iter in 1:length(annotation_model_types)) {
		model_type <- annotation_model_types[model_iter]
		model_per_snp_heritabilities <- model_data_arr[[model_iter]]$pred_tau

		per_snp_her_arr <- c(per_snp_her_arr, model_per_snp_heritabilities)
		model_arr <- c(model_arr, rep(model_type, length(model_per_snp_heritabilities)))
	}
	per_snp_her_arr[per_snp_her_arr < 0.0] = 0.0
	per_snp_her_arr = per_snp_her_arr + 1e-40

	df <- data.frame(model_per_snp_heritabilities=per_snp_her_arr, her_model=factor(model_arr, levels=annotation_model_types))

	p <- ggplot(df, aes(x=her_model, y=model_per_snp_heritabilities)) + 
  		geom_violin(trim=FALSE) +
  		figure_theme() +
  		scale_y_continuous(trans='log') +
  		theme(axis.text.x = element_text(angle = 90,hjust=1, vjust=.5, size=10)) +
  		labs(y="Expected per-snp heritability", x="heritability model", fill="")
  	return(p)
}

make_scatterplot_comparing_per_snp_log_like_of_two_models <- function(model_data_arr, annotation_model_types, model1, model2) {
	model1_index = which(annotation_model_types == model1)
	model2_index = which(annotation_model_types == model2)

	model1_log_likes <- model_data_arr[[model1_index]]$ldsc_gamma_log_like
	model2_log_likes <- model_data_arr[[model2_index]]$ldsc_gamma_log_like

	df <-data.frame(model1_log_likes=model1_log_likes, model2_log_likes=model2_log_likes)



    p <- ggplot(data=df, aes(x=model1_log_likes, y=model2_log_likes)) +
   		geom_point(alpha=.5,size=.4) +
    	figure_theme() +
    	geom_abline(color='red') +
    	labs(x=paste0(model1, " per-snp log likelihood"), y=paste0(model2, " per-snp log likelihood"))

    return(p)
}
make_histogram_comparing_difference_per_snp_log_like_of_two_models <- function(model_data_arr, annotation_model_types, model1, model2) {
	model1_index = which(annotation_model_types == model1)
	model2_index = which(annotation_model_types == model2)

	model1_log_likes <- model_data_arr[[model1_index]]$ldsc_gamma_log_like
	model2_log_likes <- model_data_arr[[model2_index]]$ldsc_gamma_log_like

	df <-data.frame(delta_log_likelihood=model1_log_likes-model2_log_likes)

     p <- ggplot(data=df, aes(x=delta_log_likelihood)) +
   		geom_histogram() +
    	figure_theme() +
    	labs(x=paste0(model1, " - ", model2, " per-snp log likelihood"))
    return(p)
}



make_scatterplot_comparing_predicted_vs_observed_chi_sq <- function(model_data_arr, annotation_model_types, model1) {
	model1_index = which(annotation_model_types == model1)

	pred_chi_sq <-  model_data_arr[[model1_index]]$pred_chi_sq
	observed_chi_sq <- model_data_arr[[model1_index]]$observed_chi_sq

	df <-data.frame(pred_chi_sq=pred_chi_sq, observed_chi_sq=observed_chi_sq)

    p <- ggplot(data=df, aes(x=pred_chi_sq, y=observed_chi_sq)) +
   		geom_point(alpha=.5,size=.5) +
    	figure_theme() +
    	geom_abline(color='red') +
    	labs(x="Predicted chi squared", y="Observed chi squared",title=model1)
    return(p)	

}

make_scatterplot_comparing_per_snp_log_like_of_two_models_colored_by_chi_sq <- function(model_data_arr, annotation_model_types, model1, model2) {
	model1_index = which(annotation_model_types == model1)
	model2_index = which(annotation_model_types == model2)

	model1_log_likes <- model_data_arr[[model1_index]]$ldsc_gamma_log_like
	model2_log_likes <- model_data_arr[[model2_index]]$ldsc_gamma_log_like
	observed_chi_sq <- model_data_arr[[model2_index]]$observed_chi_sq

	df <-data.frame(model1_log_likes=model1_log_likes, model2_log_likes=model2_log_likes, observed_chi_sq=log(observed_chi_sq))

    p <- ggplot(data=df, aes(x=model1_log_likes, y=model2_log_likes,color=observed_chi_sq)) +
   		geom_point(size=.4) +
    	figure_theme() +
    	geom_abline(color='red') +
    	labs(x=paste0(model1, " per-snp log likelihood"), y=paste0(model2, " per-snp log likelihood"))
    return(p)
}
make_scatterplot_comparing_per_snp_log_like_of_two_models_colored_by_ld_score <- function(model_data_arr, annotation_model_types, model1, model2) {
	model1_index = which(annotation_model_types == model1)
	model2_index = which(annotation_model_types == model2)

	model1_log_likes <- model_data_arr[[model1_index]]$ldsc_gamma_log_like
	model2_log_likes <- model_data_arr[[model2_index]]$ldsc_gamma_log_like
	ld_score <- model_data_arr[[model2_index]]$ld_score

	df <-data.frame(model1_log_likes=model1_log_likes, model2_log_likes=model2_log_likes, ld_score=log(ld_score))

    p <- ggplot(data=df, aes(x=model1_log_likes, y=model2_log_likes,color=ld_score)) +
   		geom_point(size=.4) +
    	figure_theme() +
    	geom_abline(color='red') +
    	labs(x=paste0(model1, " per-snp log likelihood"), y=paste0(model2, " per-snp log likelihood"))
    return(p)
}

reorganize_eval_data <- function(df) {
	df <- df[as.character(df$model) != "sldsc_linear_model",]
	df <- df[as.character(df$model) != "exp_linear_model",]

	df$model = as.character(df$model)
	df$model[df$model == "neural_network_no_drops"] = "neural_network"
	df$model[df$model == "sldsc_linear_model_non_neg_tau"] = "sldsc_linear_model"

	df$model = factor(df$model, levels=c("intercept_model", "sldsc_linear_model", "linear_model", "neural_network"))

	return(df)
}


reorganize_eval_data2 <- function(df) {
	df <- df[as.character(df$model) != "sldsc_linear_model",]

	df$model = as.character(df$model)
	df$model[df$model == "neural_network_no_drops"] = "neural_network"
	df$model[df$model == "sldsc_linear_model_non_neg_tau"] = "sldsc_linear_model"

	df$model = factor(df$model, levels=c("intercept_model", "sldsc_linear_model", "linear_model", "neural_network", "exp_linear_model"))

	return(df)
}

merge_eval_data_objects <- function(df1, df2) {
	model_arr <- c()
	sum_gamma_log_like_arr <- c()
	se_arr <- c()

	regression_snps <- c()
	regression_snps <- c(regression_snps, rep("dependent", length(df1$model)))
	regression_snps <- c(regression_snps, rep("independent", length(df1$model)))

	#print(c(rep("dependent", length(df1$model), rep("independent", length(df1$model))))
	#print(c(df1$model, df2$model))
	#print(c(df1$sum_gamma_log_like, df2$sum_gamma_log_like))

	df <- data.frame(regression_snps=regression_snps, model=c(as.character(df1$model), as.character(df2$model)), sum_gamma_log_like=c(df1$sum_gamma_log_like, df2$sum_gamma_log_like), se_sum_gamma_log_like=c(df1$se_sum_gamma_log_like, df2$se_sum_gamma_log_like))
	df$model = factor(df$model, levels=c("intercept_model", "sldsc_linear_model", "linear_model", "neural_network"))

	return(df)
}


merge_eval_data_objects3 <- function(df1, df2) {
	model_arr <- c()
	sum_gamma_log_like_arr <- c()
	se_arr <- c()


	#print(c(rep("dependent", length(df1$model), rep("independent", length(df1$model))))
	#print(c(df1$model, df2$model))
	#print(c(df1$sum_gamma_log_like, df2$sum_gamma_log_like))

	df2 = df2[(as.character(df2$model) == "neural_network") | (as.character(df2$model) == "exp_linear_model"), ]

	df2$model = as.character(df2$model)

	df2$model[1] = "neural_network_dropout_reg"

	df <- data.frame(model=c(as.character(df1$model), as.character(df2$model)), sum_gamma_log_like=c(df1$sum_gamma_log_like, df2$sum_gamma_log_like), se_sum_gamma_log_like=c(df1$se_sum_gamma_log_like, df2$se_sum_gamma_log_like))
	df$model = factor(df$model, levels=c("intercept_model", "sldsc_linear_model", "linear_model", "neural_network", "neural_network_dropout_reg", "exp_linear_model"))

	return(df)
}

merge_eval_data_objects2 <- function(df1, df2) {
	model_arr <- c()
	sum_gamma_log_like_arr <- c()
	se_arr <- c()


	#print(c(rep("dependent", length(df1$model), rep("independent", length(df1$model))))
	#print(c(df1$model, df2$model))
	#print(c(df1$sum_gamma_log_like, df2$sum_gamma_log_like))

	df2 = df2[as.character(df2$model) == "neural_network", ]

	df2$model = as.character(df2$model)

	df2$model[1] = "neural_network_dropout_reg"

	df <- data.frame(model=c(as.character(df1$model), as.character(df2$model)), sum_gamma_log_like=c(df1$sum_gamma_log_like, df2$sum_gamma_log_like), se_sum_gamma_log_like=c(df1$se_sum_gamma_log_like, df2$se_sum_gamma_log_like))
	df$model = factor(df$model, levels=c("intercept_model", "sldsc_linear_model", "linear_model", "neural_network", "neural_network_dropout_reg"))

	return(df)
}

trait_name = args[1]
non_linear_sldsc_results_dir = args[2]



eval_file <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_marginal_updates_results_training_data_1_grad_steps_False_independent_reg_snps_even_odd_testing_ld_score_regression_eval.txt")
eval_data_false_test <- reorganize_eval_data(read.table(eval_file, header=TRUE))

output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_odd_dependent_reg.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data_false_test)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=5.0, units="in")

eval_file <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_marginal_updates_results_training_data_1_grad_steps_False_independent_reg_snps_even_even_testing_ld_score_regression_eval.txt")
eval_data_false_train <- reorganize_eval_data(read.table(eval_file, header=TRUE))

output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_even_dependent_reg.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data_false_train)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=5.0, units="in")



eval_file <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_marginal_updates_results_training_data_1_grad_steps_True_independent_reg_snps_even_odd_testing_ld_score_regression_eval.txt")
eval_data_true_test <- reorganize_eval_data2(read.table(eval_file, header=TRUE))

output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_odd_independent_reg.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data_false_train)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=5.0, units="in")

eval_file <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_marginal_updates_results_training_data_1_grad_steps_True_independent_reg_snps_even_even_testing_ld_score_regression_eval.txt")
eval_data_true_train <- reorganize_eval_data2(read.table(eval_file, header=TRUE))

output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_even_independent_reg.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data_true_train)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=5.0, units="in")



eval_data_test = merge_eval_data_objects2(eval_data_false_test, eval_data_true_test)
eval_data_train = merge_eval_data_objects2(eval_data_false_train, eval_data_true_train)


output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_odd.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data_train)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=5.0, units="in")

output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_even.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data_test)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=5.0, units="in")



eval_data_test = merge_eval_data_objects3(eval_data_false_test, eval_data_true_test)
eval_data_train = merge_eval_data_objects3(eval_data_false_train, eval_data_true_train)


output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_odd_exp.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data_train)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=5.0, units="in")

output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_even_exp.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data_test)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=5.0, units="in")

#ldsc_log_like_barplot <- make_ldsc_log_like_barplot_independent_vs_dependent_regression_snps(eval_data_true_train)




print("DONE")

# INTERCEPT MODEL MUST BE LAST
model_types <- c("neural_network_no_drops", "linear_model")
annotation_model_types <- c("neural_network_no_drops", "linear_model")

model_data_arr <- list()

for (model_iter in 1:length(model_types)) {
	model_type <- model_types[model_iter]
	#model_name <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_univariate_updates_results_training_data_", model_type, "_even_odd_testing_ld_score_regression_eval_temper.txt")
	model_name <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_marginal_updates_results_training_data_", model_type, "_1_grad_steps_False_independent_reg_snps_even_even_testing_ld_score_regression_eval_temper.txt")
	df <- read.table(model_name, header=TRUE, sep="\t")
	model_data_arr[[model_iter]] = df
}
print('saved')
saveRDS(model_data_arr, "data_arr.RDS")
model_data_arr <- readRDS("data_arr.RDS")

##############################################
# Make violin plot showing expected per snp heritability
##############################################
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_heritability_violinplot.pdf")
violin_plot <- make_violin_plot_show_expected_per_snp_heritabilities(model_data_arr, annotation_model_types)
ggsave(violin_plot, file=output_file, width=7.2, height=6.0, units="in")









if (FALSE) {

##############################################
# Make Scatter plot compaing pred chi-sq vs observed chi sq
##############################################
model1 <- "neural_network"
output_file <- paste0(non_linear_sldsc_results_dir, "predicted_vs_observed_snp_chi_sq_", model1,".pdf")
scatter <- make_scatterplot_comparing_predicted_vs_observed_chi_sq(model_data_arr, model_types, model1)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")


model1 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "predicted_vs_observed_snp_chi_sq_", model1,".pdf")
scatter <- make_scatterplot_comparing_predicted_vs_observed_chi_sq(model_data_arr, model_types, model1)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

model1 <- "sldsc_linear_model_non_neg_tau"
output_file <- paste0(non_linear_sldsc_results_dir, "predicted_vs_observed_snp_chi_sq_", model1,".pdf")
scatter <- make_scatterplot_comparing_predicted_vs_observed_chi_sq(model_data_arr, model_types, model1)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")


##############################################
# Make histogram compaing per-snp log like between two models
##############################################
model1 <- "neural_network"
model2 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_difference_histogram_", model1, "_", model2,".pdf")
histo <- make_histogram_comparing_difference_per_snp_log_like_of_two_models(model_data_arr, model_types, model1, model2)
ggsave(histo, file=output_file, width=7.2, height=6.0, units="in")


##############################################
# Make Scatter plot compaing per-snp log like between two models
##############################################
model1 <- "neural_network"
model2 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_scatter_", model1, "_", model2,".pdf")
scatter <- make_scatterplot_comparing_per_snp_log_like_of_two_models(model_data_arr, model_types, model1, model2)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

model1 <- "neural_network"
model2 <- "intercept_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_scatter_", model1, "_", model2,".pdf")
scatter <- make_scatterplot_comparing_per_snp_log_like_of_two_models(model_data_arr, model_types, model1, model2)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

model1 <- "neural_network"
model2 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_scatter_", model1, "_", model2,"_colored_by_chi_sq.pdf")
scatter <- make_scatterplot_comparing_per_snp_log_like_of_two_models_colored_by_chi_sq(model_data_arr, model_types, model1, model2)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

model1 <- "neural_network"
model2 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_scatter_", model1, "_", model2,"_colored_by_ld_score.pdf")
scatter <- make_scatterplot_comparing_per_snp_log_like_of_two_models_colored_by_ld_score(model_data_arr, model_types, model1, model2)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

##############################################
# Make violin plot showing expected per snp heritability
##############################################
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_heritability_violinplot.pdf")
violin_plot <- make_violin_plot_show_expected_per_snp_heritabilities(model_data_arr, annotation_model_types)
ggsave(violin_plot, file=output_file, width=7.2, height=6.0, units="in")

##############################################
# Make violin plot showing expected chi-squared stats
##############################################
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_chi_squared_violinplot.pdf")
violin_plot <- make_violin_plot_show_expected_per_snp_chi_squared(model_data_arr, annotation_model_types)
ggsave(violin_plot, file=output_file, width=7.2, height=6.0, units="in")

output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_log_chi_squared_violinplot.pdf")
violin_plot <- make_violin_plot_show_expected_per_snp_log_chi_squared(model_data_arr, annotation_model_types)
ggsave(violin_plot, file=output_file, width=7.2, height=6.0, units="in")

##############################################
# Make violin plot showing log likelihoods
##############################################
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_violinplot.pdf")
violin_plot <- make_violin_plot_show_expected_per_snp_sldsc_log_like(model_data_arr, annotation_model_types)
ggsave(violin_plot, file=output_file, width=7.2, height=6.0, units="in")




}













if (FALSE) {

model_types <- c("sldsc_linear_model_non_neg_tau", "neural_network_no_drops", "neural_network", "linear_model", "intercept_model")
annotation_model_types <- c("sldsc_linear_model_non_neg_tau", "neural_network_no_drops", "neural_network", "linear_model")

model_data_arr <- list()

for (model_iter in 1:length(model_types)) {
	model_type <- model_types[model_iter]
	model_name <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_marginal_updates_results_training_data_", model_type, "_even_chrom_14_testing_ld_score_regression_eval_temper.txt")
	df <- read.table(model_name, header=TRUE, sep="\t")
	model_data_arr[[model_iter]] = df
}
print('saved')
saveRDS(model_data_arr, "data_train_arr.RDS")

model_data_arr <- readRDS("data_train_arr.RDS")



##############################################
# Make Scatter plot compaing pred chi-sq vs observed chi sq
##############################################
model1 <- "neural_network"
output_file <- paste0(non_linear_sldsc_results_dir, "predicted_vs_observed_snp_chi_sq_", model1,"_train.pdf")
scatter <- make_scatterplot_comparing_predicted_vs_observed_chi_sq(model_data_arr, model_types, model1)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")


model1 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "predicted_vs_observed_snp_chi_sq_", model1,"_train.pdf")
scatter <- make_scatterplot_comparing_predicted_vs_observed_chi_sq(model_data_arr, model_types, model1)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

model1 <- "sldsc_linear_model_non_neg_tau"
output_file <- paste0(non_linear_sldsc_results_dir, "predicted_vs_observed_snp_chi_sq_", model1,"_train.pdf")
scatter <- make_scatterplot_comparing_predicted_vs_observed_chi_sq(model_data_arr, model_types, model1)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

print("DONE")
##############################################
# Make Scatter plot compaing per-snp log like between two models
##############################################
model1 <- "neural_network"
model2 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_scatter_", model1, "_", model2,"_train.pdf")
scatter <- make_scatterplot_comparing_per_snp_log_like_of_two_models(model_data_arr, model_types, model1, model2)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

model1 <- "neural_network"
model2 <- "intercept_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_scatter_", model1, "_", model2,"_train.pdf")
scatter <- make_scatterplot_comparing_per_snp_log_like_of_two_models(model_data_arr, model_types, model1, model2)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

model1 <- "neural_network"
model2 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_scatter_", model1, "_", model2,"_colored_by_chi_sq_train.pdf")
scatter <- make_scatterplot_comparing_per_snp_log_like_of_two_models_colored_by_chi_sq(model_data_arr, model_types, model1, model2)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

model1 <- "neural_network"
model2 <- "linear_model"
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_scatter_", model1, "_", model2,"_colored_by_ld_score_train.pdf")
scatter <- make_scatterplot_comparing_per_snp_log_like_of_two_models_colored_by_ld_score(model_data_arr, model_types, model1, model2)
ggsave(scatter, file=output_file, width=7.2, height=6.0, units="in")

##############################################
# Make violin plot showing expected per snp heritability
##############################################
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_heritability_violinplot_train.pdf")
violin_plot <- make_violin_plot_show_expected_per_snp_heritabilities(model_data_arr, annotation_model_types)
ggsave(violin_plot, file=output_file, width=7.2, height=6.0, units="in")

##############################################
# Make violin plot showing expected chi-squared stats
##############################################
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_chi_squared_violinplot_train.pdf")
violin_plot <- make_violin_plot_show_expected_per_snp_chi_squared(model_data_arr, annotation_model_types)
ggsave(violin_plot, file=output_file, width=7.2, height=6.0, units="in")

##############################################
# Make violin plot showing log likelihoods
##############################################
output_file <- paste0(non_linear_sldsc_results_dir, "expected_per_snp_sldsc_log_like_violinplot_train.pdf")
violin_plot <- make_violin_plot_show_expected_per_snp_sldsc_log_like(model_data_arr, annotation_model_types)
ggsave(violin_plot, file=output_file, width=7.2, height=6.0, units="in")

}

