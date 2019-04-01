reticulate::use_condaenv("tfnightly", required = TRUE)
library(keras)
use_implementation("tensorflow")
library(tensorflow)
# tf_version() # 1.14
tfe_enable_eager_execution(device_policy = "silent")
library(tfprobability)
library(tidyverse)
library(recipes)
library(tfdatasets)

source("analysis/utils.R")
