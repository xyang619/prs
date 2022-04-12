#!/usr/bin/env Rscript
library(optparse)

option_list <- list(
  make_option(c("-g", "--genotype"), type="character", default=NULL, 
              help="Genotype data in plink bed/fam/bim format, required", metavar="path"),
  make_option(c("-s", "--summary-statistics"), type="character", default=NULL, 
              help="GWAS summary statistics, required", metavar="sumstats"),
  make_option(c("-p", "--prefix"), type="character", default="output", 
              help="Prefix for output files, default=output", metavar="prefix"),
  make_option(c("-c", "--cache-path"), type="character", default="cache", 
              help="Cache path for temporary files, default=cache", metavar="path"),
  make_option(c("-t", "--num-threads"), type="integer", default=0, 
              help="Number of threads to use, 0 means using all available cores, default=0", metavar="THREADS")
); 
parser <- OptionParser(option_list=option_list)
args   <- parse_args(parser)

if (is.null(args$genotype) || is.null(args$`summary-statistics`)) {
  stop("Provide arguments are not sufficient, please check again!", call.=FALSE)
}

# print(args) 

library(bigsnpr)
library(ggplot2)

if (args$`num-threads` <= 0) {
  NCORES <- nb_cores()
} else {
  NCORES <- args$`num-threads`
}

if (!file.exists(args$`cache-path`))
  dir.create(args$`cache-path`)

cached_genotype <- paste(args$genotype, ".rds", sep = "") 

if (file.exists(cached_genotype)) {
  file.remove(cached_genotype)
  file.remove(paste(args$genotype, ".bk", sep = ""))
}
snp_readBed(paste(args$genotype, ".bed", sep = ""))

obj.bigSNP <- snp_attach(cached_genotype)
G   <- obj.bigSNP$genotypes
CHR <- obj.bigSNP$map$chromosome
POS <- obj.bigSNP$map$physical.pos
POS2<- obj.bigSNP$map$genetic.dist
map <- setNames(obj.bigSNP$map[-3], c("chr", "rsid", "pos", "a1", "a0"))
y   <- obj.bigSNP$fam$affection

# Read external summary statistics
sumstats  <- bigreadr::fread2(args$`summary-statistics`)
sumstats$n_eff <- sumstats$N
df_beta <- snp_match(sumstats, map, join_by_pos = FALSE)  # use rsid instead of pos

for (chr in 1:22) {
  ## row indices in 'df_beta'
  ind.chr <- which(df_beta$chr == chr)
  ## column indices in 'G'
  ind.chr2 <- df_beta$`_NUM_ID_`[ind.chr]
  
  corr0 <- snp_cor(G, ind.col = ind.chr2, size = 3 / 1000, infos.pos = POS2[ind.chr2], ncores = NCORES)
  
  if (chr == 1) {
    ld <- Matrix::colSums(corr0^2)
  } else {
    ld <- c(ld, Matrix::colSums(corr0^2))
  }
  cor_file <- paste(args$`cache-path`, "/", args$prefix, "_corr_chr_", chr, sep = "")
  corr <- as_SFBM(corr0, cor_file, compact = TRUE)
  corr$save()
}
ld_file <- paste(args$`cache-path`, "/", args$prefix, "_ldscore.rds", sep = "")
saveRDS(ld, ld_file)
ldsc <- with(df_beta, snp_ldsc(ld, length(ld), chi2 = (beta / beta_se)^2, sample_size = n_eff, blocks = NULL))
h2_est <- ldsc[["h2"]]
h2_seq <- round(h2_est * c(0.3, 0.7, 1, 1.4), 4)
p_seq <- signif(seq_log(1e-5, 1, length.out = 21), 2)
params <- expand.grid(p = p_seq, h2 = h2_seq, sparse = c(FALSE, TRUE))

pdf(paste(args$`cache-path`, "/", args$prefix, "_par_search.pdf", sep=""), width=12, height=8)

for (chr in 1:22) {
  ind.chr <- which(df_beta$chr == chr)
  ind.chr2 <- df_beta$`_NUM_ID_`[ind.chr]
  corr <- readRDS(paste(args$`cache-path`, "/", args$prefix, "_corr_chr_", chr, ".rds", sep = ""))
  beta_grid <-snp_ldpred2_grid(corr, df_beta[ind.chr,], params, ncores = NCORES)
  pred_grid <- big_prodMat(G, beta_grid, ind.col = ind.chr2)
  params$score <- apply(pred_grid, 2, function(x) {
    if (all(is.na(x)) || sd(x) < 1e-9) return(NA)
    summary(lm(y ~ x))$coef["x", 3]
  })

  beta_file <- paste(args$`cache-path`, "/", args$prefix, "_beta_chr_", chr, ".rds", sep = "")
  par_file <- paste(args$`cache-path`, "/", args$prefix, "_par_search_chr_", chr, ".rds", sep = "")
  saveRDS(beta_grid, beta_file)
  saveRDS(params, par_file)
  if (chr == 1) {
    max_col <- which.max(params$score)
    best_pred <- pred_grid[, max_col]
    beta_new <- beta_grid[, max_col]
  } else {
    best_pred <- cbind(best_pred, pred_grid[, max_col])
    beta_new <- rbind(beta_new, beta_grid[, max_col])
  }
  sfig <- ggplot(params, aes(x = p, y = score, color = as.factor(h2))) +
    theme_bigstatsr() +
    geom_point() +
    geom_line() +
    scale_x_log10(breaks = 10^(-5:0), minor_breaks = params$p) +
    facet_wrap(~ sparse, labeller = label_both) +
    labs(y = "GLM Z-Score", color = "h2") +
    theme(legend.position = "top", panel.spacing = unit(1, "lines"))
  print(sfig)
}
saveRDS(beta_new, paste(args$`cache-path`,  "/", args$prefix, "_beta.rds", sep = ""))
saveRDS(best_pred, paste(args$`cache-path`, "/", args$prefix, "_best_pred.rds", sep = ""))
dev.off()
