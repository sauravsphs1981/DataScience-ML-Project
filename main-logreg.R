# Setup----
library(tidyverse)
library(tidymodels)
library(Hmisc)
library(doParallel)
library(doFuture)
library(ranger)
library(vip)
library(AUC)
library(caret)
library(themis)
data <- read_csv("C:/Users/saura/OneDrive/Desktop/Research/In-pt stress/ML-Clean.csv")

set.seed(42)

# Preprocess data----
data_split <- data %>% 
  initial_split(data, prop = 1/2, strata = DIED)
train_data <- data_split %>% training
test_data <- data_split %>% testing

data_recipe <- recipe(DIED ~ ., data = train_data) %>% 
  step_rm(LOS, ATYPE, chest_pain, stress_test, PCI) %>%  # dropping due to we don't want to use LOS to predict died, and the others have a single value and thus are not useful) %>% 
  update_role(key_nis, new_role="id") %>%  # exclude, but keep as an id
  add_role(RACE, TRAN_IN, TRAN_OUT, white,  # assuming these are categorical variables
           new_role="makeDummy") %>%
  add_role(AGE, DISCWT, HCUP_ED, NCHRONIC, NDX, NIS_STRATUM, NPR, TOTCHG, YEAR, HOSP_BEDSIZE, TOTAL_DISC,
           new_role="continuous") %>%
  add_role(AWEEKEND, ELECTIVE, FEMALE, HOSP_TEACH, CM_AIDS, CM_ALCOHOL, CM_ANEMDEF, CM_ARTH, CM_BLDLOSS, CM_CHF, CM_CHRNLUNG, CM_COAG, CM_DEPRESS,
           CM_DM, CM_DMCX, CM_DRUG, CM_HTN_C, CM_HYPOTHY, CM_LIVER, CM_LYMPH, CM_LYTES, CM_METS, CM_NEURO, CM_OBESE, CM_PARA, CM_PERIVASC, CM_PSYCH, CM_PULMCIRC,
           CM_RENLFAIL, CM_TUMOR, CM_ULCER, CM_VALVE, CM_WGHTLOSS, insu, teach, htn, dm, smoker, obese, chf, afib, pad, CAD, PrMI, PrPCI, PrCABG, Prstroke, prPPM,
           prICD, prDevice, anemia, coagulopathy, prPE_DVT, copd, pulm_htn, ckd, liver,
           new_role="binary") %>%
  step_mutate_at(has_role('predictor'), all_outcomes(), fn=~tidyr::replace_na(., 0)) %>%  # replace all NA with 0 [NOTE: BIG ASSUMPTION]
  step_mutate_at(has_role('makeDummy'), all_outcomes(), fn=~factor(.)) %>%
  step_dummy(has_role('makeDummy')) %>% 
  step_normalize(has_role('continuous')) %>% 
  step_pca(has_role('continuous'), threshold=0.9) %>% 
  step_rose(DIED)  # oversample

# Bake data and try out RFE----
# baked_train_data <- data_recipe %>% 
#   prep() %>% 
#   bake(train_data)
# 
# library(randomForest)
# control <- rfeControl(functions=rfFuncs, method="cv", number=3)
# rfe_results <- rfe(baked_train_data %>% select(-key_nis, -DIED), 
#                    baked_train_data %>% select(DIED) %>% unlist,
#                    rfeControl=control,
#                    metric = 'Kappa')

# Model Selection----
cross_validation_scheme <- vfold_cv(train_data, v = 3, repeats = 1, strata = DIED)

# Add parallel processing for speed----
# registerDoFuture()
# plan(multicore, workers=15)

# Run model convenience function----
fit_model <- function(model, recipe, grid, cv_scheme) {
  data_workflow <- workflow() %>% 
    add_model(model) %>% 
    add_recipe(recipe)
  
  results <- data_workflow %>% 
    tune_grid(cv_scheme,
              grid = grid,
              control = control_grid(save_pred = TRUE),
              metrics = metric_set(roc_auc))
  
  return(results)
}

plot_model <- function(results, hyperparameter) {
  plot <- results %>% 
    collect_metrics() %>% 
    ggplot(aes(x = !!as.name(hyperparameter), y = mean)) + 
    geom_point() + 
    geom_line() + 
    ylab("Area under the ROC Curve") +
    scale_x_log10(labels = scales::label_number())
  return(plot)
}

# Set up models for fitting----
GRID_SIZE <- 15

lasso_model <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine('glmnet')
lasso_grid <- grid_random(parameters(lasso_model), size = GRID_SIZE)

ridge_model <- logistic_reg(penalty = tune(), mixture = 0) %>%
  set_engine('glmnet')
ridge_grid <- grid_random(parameters(ridge_model), size = GRID_SIZE)

elastic_net_model <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine('glmnet')
elastic_net_grid <- grid_random(parameters(elastic_net_model), size = GRID_SIZE)

rf_model <- rand_forest(trees = tune()) %>% 
  set_engine('ranger', importance = 'permutation') %>% 
  set_mode('classification')
rf_grid <- grid_random(parameters(rf_model, size = GRID_SIZE))

# Fit models to help select hyperparameters----
lasso <- fit_model(lasso_model, data_recipe, lasso_grid, cross_validation_scheme)
plot_model(lasso, 'penalty')

ridge <- fit_model(ridge_model, data_recipe, ridge_grid, cross_validation_scheme)
plot_model(ridge, 'penalty')

elastic_net <- fit_model(elastic_net_model, data_recipe, elastic_net_grid, cross_validation_scheme)
plot_model(elastic_net, 'penalty')

rf <- fit_model(rf_model, data_recipe, rf_grid, cross_validation_scheme)
plot_model(rf, 'trees')

# Traditional logistic regression----
traditional_logistic_regression_model <- logistic_reg(penalty = 0, mixture = 1) %>%
  set_engine('glmnet')

traditional_lr_wf <- workflow() %>%
  add_recipe(data_recipe) %>%
  add_model(traditional_logistic_regression_model)

traditional_lr_results <- traditional_lr_wf %>%
  last_fit(data_split)

traditional_lr_results %>%
  collect_metrics()

# Collect information on the best version of each model----
best_lasso <- select_best(lasso, 'roc_auc')
best_ridge <- select_best(ridge, 'roc_auc')
best_elastic <- select_best(elastic_net, 'roc_auc')
best_rf <- select_best(rf, 'roc_auc')

model_selection_summary <- data.frame(model = c('lasso', 'ridge', 'elastic', 'rf', 'traditional_logistic_regression'),
                                      auc = c(max(collect_metrics(lasso)$mean),
                                              max(collect_metrics(ridge)$mean),
                                              max(collect_metrics(elastic_net)$mean),
                                              max(collect_metrics(rf)$mean),
                                              collect_metrics(traditional_lr_results)$.estimate[2]),
                                      best_penalty = c(best_lasso$penalty,
                                                       best_ridge$penalty,
                                                       best_elastic$penalty,
                                                       NA,
                                                       NA),
                                      best_mixture = c(NA,
                                                       NA,
                                                       best_elastic$mixture,
                                                       NA,
                                                       NA),
                                      best_trees = c(NA,
                                                     NA,
                                                     NA,
                                                     best_rf$trees,
                                                     NA)
)

# Test fit final models----
getCoefsForLinearModel <- function(base_model, best_hyperparameters, data_recipe, train_data) {
  final_model <- finalize_model(base_model, best_hyperparameters)
  
  final_wf <- workflow() %>% 
    add_recipe(data_recipe) %>% 
    add_model(final_model)
  
  final_fitted_model <- final_wf %>% 
    fit(train_data) %>% 
    pull_workflow_fit()
  
  coefs <- final_fitted_model %>% 
    pluck('fit') %>% 
    coef(s = best$penalty)
  
  return(coefs)
}

lasso_coefs <- getCoefsForLinearModel(lasso_model, best_lasso, data_recipe, train_data)
ridge_coefs <- getCoefsForLinearModel(ridge_model, best_ridge, data_recipe, train_data)
elastic_net_coefs <- getCoefsForLinearModel(elastic_net_model, best_elastic, data_recipe, train_data)

# Fit final model on full training set using best hyperparameters----
best_auc <- select_best(rf, "roc_auc")
final_rf <- finalize_model(rf_model, best_auc)

final_rf_wf <- workflow() %>% 
  add_recipe(data_recipe) %>% 
  add_model(final_rf)

final_rf_results <- final_rf_wf %>% 
  last_fit(data_split)

final_rf_results %>% 
  collect_metrics()

final_rf_model <- final_rf_wf %>% 
  fit(train_data) %>% 
  pull_workflow_fit()

# Plot variable importance
final_rf_model %>% vip(geom = 'point')

# Plot AUC curve for traditional lr
traditional_lr_predictions <- traditional_lr_results %>% collect_predictions()
plot(AUC::sensitivity(traditional_lr_predictions$.pred_1, traditional_lr_predictions$DIED))
plot(AUC::specificity(traditional_lr_predictions$.pred_1, traditional_lr_predictions$DIED))
plot(AUC::accuracy(traditional_lr_predictions$.pred_1, traditional_lr_predictions$DIED))
plot(AUC::roc(traditional_lr_predictions$.pred_1, traditional_lr_predictions$DIED))

# Plot AUC curve----
final_rf_predictions <- final_rf_results %>% collect_predictions()
plot(AUC::sensitivity(final_rf_predictions$.pred_1, final_rf_predictions$DIED))
plot(AUC::specificity(final_rf_predictions$.pred_1, final_rf_predictions$DIED))
plot(AUC::accuracy(final_rf_predictions$.pred_1, final_rf_predictions$DIED))
plot(AUC::roc(final_rf_predictions$.pred_1, final_rf_predictions$DIED))

# Confusion matrix----
table(final_rf_predictions$.pred_class, final_rf_predictions$DIED)

