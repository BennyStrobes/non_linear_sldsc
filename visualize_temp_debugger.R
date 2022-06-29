args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(hash)

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}


make_ldsc_log_like_barplot <- function(df) {
    p <- ggplot(data=df, aes(x=model, y=sum_gamma_log_like)) +
    geom_point(stat="identity") +
    figure_theme() +
    theme(legend.position="top") +
    theme(axis.text.x = element_text(angle = 90,hjust=1, vjust=.5, size=10)) +
    labs(y="Log LDSC gamma loss", x="model", fill="") +
    geom_errorbar(aes(ymin=sum_gamma_log_like-(se_sum_gamma_log_like), ymax=sum_gamma_log_like+(se_sum_gamma_log_like)), position = position_dodge(), width = .75, size=.2)

    return(p)
}




trait_name = args[1]
non_linear_sldsc_results_dir = args[2]



eval_file <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_multivariate_results_training_data_even_odd_testing_ld_score_regression_eval.txt")

eval_data <- read.table(eval_file, header=TRUE)


output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_odd.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=6.0, units="in")


eval_file <- paste0(non_linear_sldsc_results_dir, trait_name, "_nonlinear_sldsc_multivariate_results_training_data_even_even_testing_ld_score_regression_eval.txt")

eval_data <- read.table(eval_file, header=TRUE)


output_file <- paste0(non_linear_sldsc_results_dir, "ldsc_log_like_barplot_even_even.pdf")
ldsc_log_like_barplot <- make_ldsc_log_like_barplot(eval_data)
ggsave(ldsc_log_like_barplot, file=output_file, width=7.2, height=6.0, units="in")


