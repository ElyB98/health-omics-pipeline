# scripts/simulate_rnaseq.R
dir.create("data/simulated_rnaseq", showWarnings = FALSE, recursive = TRUE)

set.seed(42)

for (i in 1:100) {
  sample_name <- paste0("sample_", i)
  lines <- c(
    paste0("@read_", sample_name, "_1"),
    paste0(paste(sample(c("A", "T", "G", "C"), 50, replace = TRUE), collapse = "")),
    "+",
    paste(rep("I", 50), collapse = "")
  )
  writeLines(lines, paste0("data/simulated_rnaseq/", sample_name, ".fq"))
}
