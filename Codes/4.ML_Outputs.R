# Analyze Multiple Models

library(caret)
library(dplyr)
library(tidyr)
library(slider)
library(readxl)
library(janitor)
library(data.table)

set.seed(14)

source('E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/Scripts/PTZ_Model_Functions.R') 

current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))

# Models to evaluate
model_names <- c("RF", "SVM", "XGBoost", "kNN", "MLP-ML")
model_files <- c("model_rf.rds", "model_svmRadial.rds", "model_xgbTree.rds", "model_knn.rds", "model_mlpML.rds")
model_paths <- paste0("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/Saved_Models/", model_files)
models <- setNames(lapply(model_paths, readRDS), model_names)

# Load filenames
l.filenames.clean <- list.files('E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/Data/Correct_Format/', pattern='*.csv')
l.filenames.id <- sub("\\.csv$", "", l.filenames.clean)

# Load sliding window data
l.sliding.window <- readRDS('E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/l.df.all_PTZ_15_PTZonly.rds')

# Load posture
l.posture <- Sys.glob("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/Data/Posture/*.csv")
l.df.posture <- lapply(l.posture, function(path) {
  df <- tryCatch(read.csv(path), error = function(e) NULL)
  if (!is.null(df) && ncol(df) >= 1 && nrow(df) > 1) {
    df <- df[-1, , drop = FALSE]
    if (ncol(df) >= 1) colnames(df)[1] <- "Posture"
    return(df)
  } else {
    return(NULL)
  }
})

# Extract velocity
abs_trunk_vel_x_list <- lapply(l.sliding.window, function(df) df$abs.trunk.vel.x)
id.table <- read_excel("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/ID_PTZ.xlsx")

# Run analysis model by model
model_name <- "kNN"  # Example: change to any of the model_names
model <- models[[model_name]]
cat("Running:", model_name, "...\n")

# Predict behavior
l.result <- lapply(l.sliding.window, function(x) {
  pred_type <- if (inherits(model, "train")) "raw" else "response"
  data.frame(behavior.label = predict(model, x[, 2:44], type = pred_type))
})

# Combine with posture and velocity
l.df.combined <- mapply(function(posture_df, result_df, vel_x) {
  n <- min(nrow(posture_df), nrow(result_df), length(vel_x))
  bind_cols(
    posture_df[seq_len(n), , drop = FALSE],
    result_df[seq_len(n), , drop = FALSE],
    data.frame(abs.trunk.vel.x = vel_x[seq_len(n)])
  )
}, l.df.posture, l.result, abs_trunk_vel_x_list, SIMPLIFY = FALSE)

l.df.combined <- lapply(l.df.combined, function(df) {
  df %>%
    mutate(
    #  behavior = ifelse(Posture == 1 & as.character(behavior.label) %in% c("IMO"), "TO", as.character(behavior.label)),
     # behavior = ifelse(Posture == 0 & as.character(behavior.label) %in% c("TO"), "HYPO", as.character(behavior.label)),
    #  behavior = ifelse(abs.trunk.vel.x < 0.75 & as.character(behavior.label) == "CL", "HYPE", behavior)
    )
})

l.df.combined <- lapply(l.df.combined, function(x) {
  x$behavior <- as.character(unlist(slide(x$behavior, getmode, .before = 7, .after = 7)))
  x
})

# Frequency of behavior
l.result.percentage <- lapply(l.df.combined, function(x) {
  if (length(x$behavior) == 0) return("No data available")
  prop.table(table(x$behavior)) * 100
})

l.result.percentage.labeled <- mapply(function(df, id) {
  df <- as.data.frame(df)
  df$ID <- id
  df
}, l.result.percentage, l.filenames.id, SIMPLIFY = FALSE)

df.results.labeled <- dplyr::bind_rows(lapply(l.result.percentage.labeled,function(x) as.data.frame((x),stringsAsFactors=FALSE)))
df.results.labeled[is.na(df.results.labeled)] <- 0
df.results.labeled <- dplyr::inner_join(df.results.labeled, id.table, by='ID')

# Fix behavior label column name
colnames(df.results.labeled)[colnames(df.results.labeled) == "Var1"] <- "Behavior"
df.results.labeled <- pivot_wider(df.results.labeled, names_from = Behavior, values_from = Freq, values_fill = 0)
df.results.labeled$Normal.Swimming <- df.results.labeled$NT + df.results.labeled$SW


# Save combined
result_folder <- paste0("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/Data/Results/", model_name, "/")
dir.create(result_folder, recursive = TRUE, showWarnings = FALSE)
mapply(function(x, y) {
  file_path <- paste0(result_folder, sub("\\.csv$", "", x), ".csv")
  write.csv(y, file_path, row.names = FALSE)
}, l.filenames.clean, l.df.combined)

write.csv(df.results.labeled, paste0("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/Data/Results_PTZ_Predictions_", model_name, ".csv"), row.names = FALSE)