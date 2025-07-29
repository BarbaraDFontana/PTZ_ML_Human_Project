# Load required libraries
library(caret)
library(pROC)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(doParallel)
library(RColorBrewer)

# Enable parallel processing
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

set.seed(14)

# Load training and testing data
df.training.allB <- readRDS('E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/df_training_15.rds')
testing_data <- readRDS('E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/df_testing_15.rds')

# Preprocess
df.training.allB$behavior.label <- as.factor(df.training.allB$behavior.label)
testing_data$behavior.label <- as.factor(testing_data$behavior.label)
df.training.allB <- df.training.allB[-1]  # remove frame column
testing_data_x <- testing_data[, -1]     # features only
testing_data_y <- testing_data$behavior.label

# Define model list
model_list <- c("rf", "svmRadial", "xgbTree", "knn", "mlpML")
model_labels <- c("Random Forest", "SVM", "XGBoost", "k-NN", "MLP-ML")

# Define optimized tuning grids
custom_grids <- list(
  rf = expand.grid(mtry = 10:20),
  svmRadial = expand.grid(C = c(10, 50, 100, 200, 400), sigma = c(0.5, 1,2,4)),
  xgbTree = expand.grid(
    nrounds = c(700, 900),
    max_depth = c(15, 18),
    eta = c(0.05, 0.1),
    gamma = c(0, 1),               # add regularization
    colsample_bytree = c(0.8, 1),
    min_child_weight = c(3, 5),
    subsample = c(0.8, 1)
  ),
  knn = expand.grid(k = seq(3, 11, 2)),
  mlpML = expand.grid(
    layer1 = c(5, 10, 15, 20),
    layer2 = c(0, 5, 10),       # test both with and without a second layer
    layer3 = c(0, 5)            # test some deeper networks
  )
)

# Define train control
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = "final")

# Create empty lists to store results
model_results <- list()
metrics_all <- list()
plots_list <- list()
training_times <- data.frame(Model = character(), Minutes = numeric(), stringsAsFactors = FALSE)

# Train and Save Models (only if missing)
for (i in seq_along(model_list)) {
  method <- model_list[i]
  label <- model_labels[i]
  model_path <- paste0("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/model_", method, ".rds")
  
  if (!file.exists(model_path)) {
    cat("Training", label, "...\n")
    start_time <- Sys.time()
    
    model <- tryCatch({
      if (method == "rf") {
        train(
          preProcess = c("center", "scale"),
          behavior.label ~ .,
          data = df.training.allB,
          method = method,
          trControl = train_control,
          tuneGrid = custom_grids[[method]],
          ntree = 1000
        )
      } else {
        train(
          preProcess = c("center", "scale"),
          behavior.label ~ .,
          data = df.training.allB,
          method = method,
          trControl = train_control,
          tuneGrid = custom_grids[[method]]
        )
      }
    }, error = function(e) {
      cat("Error training", label, ":", conditionMessage(e), "\n")
      return(NULL)
    })
    
    end_time <- Sys.time()
    duration <- round(difftime(end_time, start_time, units = "mins"), 2)
    cat("Training completed for", label, "in", duration, "minutes\n")
    training_times <- rbind(training_times, data.frame(Model = label, Minutes = duration))
    
    if (!is.null(model)) {
      saveRDS(model, model_path)
      model_results[[label]] <- model
    }
  } else {
    cat("Model already exists:", label, "— skipping training.\n")
    model_results[[label]] <- readRDS(model_path)
  }
}

# Save training durations
write.csv(training_times, "E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/model_training_times.csv", row.names = FALSE)

# ---- Ensure All Models Are Loaded ----
for (i in seq_along(model_list)) {
  method <- model_list[i]
  label <- model_labels[i]
  if (is.null(model_results[[label]])) {
    model_path <- paste0("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/model_", method, ".rds")
    if (file.exists(model_path)) {
      cat("Reloading model for analysis:", label, "\n")
      model_results[[label]] <- readRDS(model_path)
    } else {
      cat("Model missing for analysis:", label, "\n")
    }
  }
}


# ---- Analyze All Models ----
for (label in names(model_results)) {
  model <- model_results[[label]]
  pred <- predict(model, testing_data_x)
  prob <- predict(model, testing_data_x, type = "prob")
  cm <- confusionMatrix(pred, testing_data_y)
  
  cm_tab <- table(testing_data_y, pred)
  cm_prop <- cm_tab / rowSums(cm_tab)
  cm_df <- as.data.frame(cm_prop)
  colnames(cm_df) <- c("Actual", "Predicted", "Freq")
  
  p_cm <- ggplot(cm_df, aes(x = Actual, y = Predicted)) +
    geom_tile(aes(fill = Freq * 100), color = "black") +
    geom_text(aes(label = paste0(round(Freq * 100, 1), "%")), size = 5) +
    scale_fill_gradient(low = "white", high = "#1f77b4", limits = c(0, 100)) +
    labs(title = paste("Confusion Matrix:", label), x = "Actual", y = "Predicted") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  plots_list[[label]] <- p_cm
  
  class_levels <- levels(testing_data_y)
  metrics_df <- data.frame(
    Model = label,
    Class = class_levels,
    Sensitivity = cm$byClass[, "Sensitivity"],
    Specificity = cm$byClass[, "Specificity"],
    F1_Score = cm$byClass[, "F1"],
    MCC = NA,
    AUC = NA
  )
  
  for (class in class_levels) {
    binary_actual <- factor(ifelse(testing_data_y == class, "Positive", "Negative"), levels = c("Negative", "Positive"))
    roc_obj <- roc(binary_actual, prob[[class]])
    metrics_df[metrics_df$Class == class, "AUC"] <- auc(roc_obj)
    
    cm_bin <- table(binary_actual, prob[[class]] > 0.5)
    if (all(dim(cm_bin) == c(2, 2))) {
      tp <- cm_bin[2, 2]
      tn <- cm_bin[1, 1]
      fp <- cm_bin[1, 2]
      fn <- cm_bin[2, 1]
      
      numerator <- as.numeric(tp * tn - fp * fn)
      denominator <- sqrt(as.numeric(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
      mcc <- ifelse(!is.na(denominator) && denominator > 0, numerator / denominator, 0)
      metrics_df[metrics_df$Class == class, "MCC"] <- mcc
    } else {
      metrics_df[metrics_df$Class == class, "MCC"] <- NA
    }
  }
  
  metrics_all[[label]] <- metrics_df
}

final_metrics <- do.call(rbind, metrics_all)
write.csv(final_metrics, "E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/final_model_metrics.csv", row.names = FALSE)

# Reorder Model and Class factors for plotting
final_metrics$Model <- factor(final_metrics$Model, levels = c("Random Forest", "SVM", "XGBoost", "k-NN", "MLP-ML"))
final_metrics$Class <- factor(final_metrics$Class, levels = c("NT", "SW", "HYPE", "CL", "TO", "IM"))
write.csv(final_metrics, "E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/final_model_metrics.csv", row.names = FALSE)

f1_plot <- ggplot(final_metrics, aes(x = Model, y = F1_Score, fill = Class)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Purples") +
  theme_minimal() +
  labs(title = "F1 Score per Model and Class") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/F1_Score_Barplot.pdf", plot = f1_plot)

pdf("E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/Confusion_Matrices_All_Models.pdf", width = 14, height = 10)
do.call("grid.arrange", c(plots_list, ncol = 2))
dev.off()

summary_df <- final_metrics %>%
  group_by(Model) %>%
  summarise(
    Avg_F1 = mean(F1_Score, na.rm = TRUE),
    Avg_AUC = mean(AUC, na.rm = TRUE),
    Avg_MCC = mean(MCC, na.rm = TRUE)
  )
write.csv(summary_df, "E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/model_summary_averages.csv", row.names = FALSE)
print(summary_df)

# Stop parallel cluster
stopCluster(cl)


# ---- Extract and Save Best Hyperparameters ----
best_params <- lapply(model_results, function(model) model$bestTune)
best_params_df <- bind_rows(best_params, .id = "Model")

# Ensure consistent naming
best_params_df$Model <- factor(best_params_df$Model, levels = c("Random Forest", "SVM", "XGBoost", "k-NN", "MLP-ML"))

# Save to CSV
write.csv(best_params_df, "E:/Barbara/Postdoc_Denis(2024-2025)/PTZ_Projeto/ML_Human/best_hyperparameters.csv", row.names = FALSE)

# Optional: print for quick view
print(best_params_df)
