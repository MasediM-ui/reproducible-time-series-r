# ============================================================
# ADVANCED MODERN TIME SERIES IN R
# Structural, Causal, Classical vs ML
# 20 years | Monthly | Fully reproducible
# ============================================================

library(tidyverse)
library(tsibble)
library(fable)
library(feasts)
library(lubridate)
library(slider)
library(randomForest)

# -----------------------------
# 1. Simulate 20 Years of Data
# -----------------------------
set.seed(2025)

n <- 20 * 12
dates <- yearmonth(seq.Date(
  from = as.Date("2005-01-01"),
  by   = "month",
  length.out = n
))

trend <- seq(100, 180, length.out = n)
seasonality <- 10 * sin(2 * pi * (1:n) / 12)

# Intervention at Jan 2015
intervention <- ifelse(dates >= yearmonth("2015 Jan"), 1, 0)

# External regressor (e.g. economic index)
xreg <- 50 + 0.3 * (1:n) + rnorm(n, 0, 3)

noise <- rnorm(n, 0, 5)

value <- trend + seasonality + 15 * intervention + 0.5 * xreg + noise

ts_data <- tibble(
  date = dates,
  value = value,
  intervention = intervention,
  xreg = xreg
) %>%
  as_tsibble(index = date)

# -----------------------------
# 2. Structural Decomposition
# -----------------------------
ts_data %>%
  model(STL(value ~ trend(window = 15))) %>%
  components() %>%
  autoplot()

# -----------------------------
# 3. Interrupted Time Series (Causal)
# -----------------------------
its_model <- ts_data %>%
  model(
    ITS = ARIMA(value ~ intervention)
  )

report(its_model)

# -----------------------------
# Interrupted Time Series (Correct)
# -----------------------------

# Define intervention point
intervention_date <- yearmonth("2015 Jan")

# Pre-intervention data only
pre_data <- ts_data %>%
  filter(date < intervention_date)

# Fit ITS model on pre-intervention period
its_model <- pre_data %>%
  model(
    ITS = ARIMA(value ~ intervention)
  )

report(its_model)

# Create future data (post-intervention period)
future_data <- ts_data %>%
  filter(date >= intervention_date) %>%
  mutate(intervention = 0) %>%   # counterfactual: no intervention
  select(date, intervention, xreg)

# Counterfactual forecast
counterfactual <- its_model %>%
  forecast(new_data = future_data)

# Plot observed vs counterfactual
autoplot(counterfactual, ts_data) +
  labs(title = "Interrupted Time Series Counterfactual")


# -----------------------------
# 4. ARIMAX (External Regressor)
# -----------------------------
arimax_model <- ts_data %>%
  model(
    ARIMAX = ARIMA(value ~ xreg)
  )

report(arimax_model)

# -----------------------------
# 5. Classical Benchmark Models
# -----------------------------
classic_models <- ts_data %>%
  model(
    ARIMA = ARIMA(
      value,
      stepwise = TRUE,     # reduce search space
      approximation = TRUE # faster likelihood
    ),
    ETS = ETS(value)
  )

report(classic_models)
# -----------------------------
# 6. Rolling-Origin Evaluation
# -----------------------------
accuracy_classic <- ts_data %>%
  stretch_tsibble(.init = 120, .step = 12) %>%
  model(
    ARIMA = ARIMA(value),
    ETS   = ETS(value),
    ARIMAX = ARIMA(value ~ xreg)
  ) %>%
  accuracy()

accuracy_classic

# -----------------------------
# 7. Machine Learning Baseline
# (Lag-based Random Forest)
# -----------------------------
ml_data <- ts_data %>%
  mutate(
    lag1 = lag(value, 1),
    lag12 = lag(value, 12),
    roll_mean_6 = slide_dbl(value, mean, .before = 5, .complete = TRUE)
  ) %>%
  drop_na() %>%
  as_tibble()

train <- ml_data %>% filter(date < yearmonth("2020 Jan"))
test  <- ml_data %>% filter(date >= yearmonth("2020 Jan"))

rf_model <- randomForest(
  value ~ lag1 + lag12 + roll_mean_6 + xreg,
  data = train
)

test <- test %>%
  mutate(
    rf_pred = predict(rf_model, newdata = test)
  )

# ML accuracy
ml_accuracy <- tibble(
  RMSE = sqrt(mean((test$value - test$rf_pred)^2)),
  MAE  = mean(abs(test$value - test$rf_pred))
)

ml_accuracy

# -----------------------------
# 8. Final Forecast (Preferred Model)
# -----------------------------
final_model <- ts_data %>%
  model(ARIMAX = ARIMA(value ~ xreg))

future_xreg <- tibble(
  date = yearmonth(seq.Date(
    from = as.Date("2025-01-01"),
    by   = "month",
    length.out = 36
  )),
  xreg = mean(ts_data$xreg)
) %>%
  as_tsibble(index = date)

final_fc <- final_model %>%
  forecast(new_data = future_xreg)

autoplot(final_fc, ts_data)
