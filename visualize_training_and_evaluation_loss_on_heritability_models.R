args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(hash)

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}





load_in_training_and_evaluation_data <- function(training_chrom, eval_chrom, itera, trait_name, models, input_dir, thresh) {
	model_arr <- c()
	train_test_arr <- c()
	iteration_arr <- c()
	accuracy_arr <- c()

	for (model_iter in 1:length(models)) {
		model_name <- models[model_iter]

		training_file <- paste0(input_dir, trait_name, "_nonlinear_sldsc_marginal_updates_gamma_likelihood_results_training_data_", model_name, "_fixed_intercept_in_sample_ld_", training_chrom, "_train_", eval_chrom, "_eval_annotations_to_gamma_model_", itera, "_training_loss.txt")
		eval_file <- paste0(input_dir, trait_name, "_nonlinear_sldsc_marginal_updates_gamma_likelihood_results_training_data_", model_name, "_fixed_intercept_in_sample_ld_", training_chrom, "_train_", eval_chrom, "_eval_annotations_to_gamma_model_", itera, "_evaluation_loss.txt")
		temp_data <- (read.table(training_file, header=FALSE))$V1
		N <- length(temp_data)
		temp_data[temp_data > thresh] = thresh
		accuracy_arr <- c(accuracy_arr, temp_data)
		model_arr <- c(model_arr, rep(model_name, N))
		train_test_arr <- c(train_test_arr, rep("Train", N))
		iteration_arr <- c(iteration_arr, seq(0,500,5))
		temp_data <- (read.table(eval_file, header=FALSE))$V1
		temp_data[temp_data > thresh] = thresh
		N <- length(temp_data)
		accuracy_arr <- c(accuracy_arr, temp_data)
		model_arr <- c(model_arr, rep(model_name, N))
		train_test_arr <- c(train_test_arr, rep("Test", N))
		iteration_arr <- c(iteration_arr, seq(0,500,5))
	}
	df <- data.frame(model=factor(model_arr, levels=models), train_test=factor(train_test_arr, levels=c("Train", "Test")), iteration=iteration_arr, loss=accuracy_arr)
	return(df)
}

make_train_test_line_plot <- function(df, train_test, trait, chrom_num) {
	p <- ggplot(df, aes(x=iteration, y=loss, group=model))+ 
 		 geom_line(aes(color=model))+
   	 	theme(legend.position="bottom") +
    	labs(title=paste0(train_test, " / ", trait, " / ", chrom_num),x="Training iteration", y = "Loss")+
   	 	figure_theme() +
   	 	guides(colour = guide_legend(nrow = 4))
   	 return(p)
}




input_dir <- args[1]
visualization_dir <- args[2]



training_chrom = "all"
eval_chrom = "chr_14"
itera = "500"
trait_name = "blood_WHITE_COUNT"
models <- c("intercept_model", "linear_model", "exp_linear_model", "neural_network_no_drops_scale", "reduced_dimension_neural_network_model", "neural_network_10", "neural_network_20", "neural_network_30")
df <- load_in_training_and_evaluation_data(training_chrom, eval_chrom, itera, trait_name, models, input_dir, .73)

line_plot_train <- make_train_test_line_plot(df[as.character(df$train_test) =="Train",] , "Train", trait_name, eval_chrom)
line_plot_test <- make_train_test_line_plot(df[as.character(df$train_test) =="Test",], "Test", trait_name, eval_chrom)

output_file <- paste0(visualization_dir, "train_loss_line_plot_", trait_name, "_", training_chrom, "_", eval_chrom, ".pdf")
ggsave(line_plot_train, file=output_file, width=7.2, height=4.5, units="in")
output_file <- paste0(visualization_dir, "test_loss_line_plot_", trait_name, "_", training_chrom, "_", eval_chrom, ".pdf")
ggsave(line_plot_test, file=output_file, width=7.2, height=4.5, units="in")

training_chrom = "all"
eval_chrom = "chr_15"
itera = "500"
trait_name = "blood_WHITE_COUNT"
models <- c("intercept_model", "linear_model", "exp_linear_model", "neural_network_no_drops", "reduced_dimension_neural_network_model", "neural_network_10", "neural_network_20", "neural_network_30")
df <- load_in_training_and_evaluation_data(training_chrom, eval_chrom, itera, trait_name, models, input_dir, .73)

line_plot_train <- make_train_test_line_plot(df[as.character(df$train_test) =="Train",] , "Train", trait_name, eval_chrom)
line_plot_test <- make_train_test_line_plot(df[as.character(df$train_test) =="Test",], "Test", trait_name, eval_chrom)

output_file <- paste0(visualization_dir, "train_loss_line_plot_", trait_name, "_", training_chrom, "_", eval_chrom, ".pdf")
ggsave(line_plot_train, file=output_file, width=7.2, height=4.5, units="in")
output_file <- paste0(visualization_dir, "test_loss_line_plot_", trait_name, "_", training_chrom, "_", eval_chrom, ".pdf")
ggsave(line_plot_test, file=output_file, width=7.2, height=4.5, units="in")
print("DONE")
training_chrom = "all"
eval_chrom = "chr_15"
itera = "500"
trait_name = "body_HEIGHTz"
models <- c("intercept_model", "linear_model", "exp_linear_model", "neural_network_no_drops", "reduced_dimension_neural_network_model", "neural_network_10", "neural_network_20", "neural_network_30")
df <- load_in_training_and_evaluation_data(training_chrom, eval_chrom, itera, trait_name, models, input_dir, 2.2)

line_plot_train <- make_train_test_line_plot(df[as.character(df$train_test) =="Train",] , "Train", trait_name, eval_chrom)
line_plot_test <- make_train_test_line_plot(df[as.character(df$train_test) =="Test",], "Test", trait_name, eval_chrom)

output_file <- paste0(visualization_dir, "train_loss_line_plot_", trait_name, "_", training_chrom, "_", eval_chrom, ".pdf")
ggsave(line_plot_train, file=output_file, width=7.2, height=4.5, units="in")
output_file <- paste0(visualization_dir, "test_loss_line_plot_", trait_name, "_", training_chrom, "_", eval_chrom, ".pdf")
ggsave(line_plot_test, file=output_file, width=7.2, height=4.5, units="in")




