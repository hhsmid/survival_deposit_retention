library(LTRCforests)
library(survival)
library(arrow)
library(dplyr)
library(magrittr)
library(data.table)
library(ggplot2)
library(rms)
library(openxlsx)
library(parallel)
library(xtable)

############################################################################################################
############################################################################################################

shat_funct <- function(Ni, data, pred = NULL, tpnt, obj.roc = NULL){
  ## This function is to compute the estimated survival probability of the Ni-th subject
  id_uniq <- unique(data$ID)
  
  ## the i-th data
  TestData <- data[data$ID == id_uniq[Ni], ]
  
  TestT <- c(TestData[1, "Start"], TestData[, "Stop"])
  TestTIntN <- nrow(TestData)
  
  tpntL <- c(TestT, tpnt)
  torder <- order(tpntL)
  tpntLod <- tpntL[torder]
  tlen <- length(tpntLod)
  
  
  if (is.null(obj.roc)) {
    if ("survfit.cox" %in% class(pred) || "survfitcox" %in% class(pred)){
      ## Compute the estimated survival probability of the Ni-th subject
      Shat_temp <- matrix(0, nrow = 1, ncol = tlen)
      
      r.ID <- findInterval(tpntLod, TestT)
      r.ID[r.ID > TestTIntN] <- TestTIntN
      
      jall <- unique(r.ID[r.ID > 0])
      nj <- length(jall)
      
      ## Deal with left-truncation
      Shat_temp[1, r.ID == 0] <- 1
      
      if(nj == 1){
        ## Get the index of the Pred to compute Shat
        II = which(data$ID == id_uniq[Ni])[jall[nj]]
        Shat_i = ipred::getsurv(pred[II], tpntLod[r.ID == jall[nj]])
        Shat_temp[1, r.ID == jall[nj]] <- Shat_i / Shat_i[1]
      } else {
        ShatR_temp <- matrix(0, nrow = 1, ncol = nj + 1)
        ShatR_temp[1, 1] = 1
        
        # S_1(L_1), S_2(L_2), S_3(L_3), ..., S_{nj}(L_{nj})
        qL = rep(0, nj)
        for (j in 1:nj){
          ## Get the index of the Pred to compute Shat
          II <- which(data$ID == id_uniq[Ni])[1] + jall[j] - 1
          Shat_j = ipred::getsurv(pred[II], tpntLod[r.ID == jall[j]])
          
          qL[j] <- Shat_j[1]
          # S_{j}(R_{j}), j=1,...nj-1
          jR = ipred::getsurv(pred[II], TestT[j + 1])
          ShatR_temp[1, j + 1] = jR / qL[j]
          Shat_temp[1, r.ID == jall[j]] <- Shat_j / qL[j]
        }
        
        ql0 <- which(qL == 0)
        if (length(ql0) > 0){
          if (any(qL > 0)){
            maxqlnot0 <- max(which(qL > 0))
            
            ql0lmax <- ql0[ql0 < maxqlnot0]
            ql0mmax <- ql0[ql0 >= maxqlnot0]
            ShatR_temp[1, ql0lmax + 1] <- 1
            Shat_temp[1, r.ID %in% jall[ql0lmax]] <- 1
            ShatR_temp[1, ql0mmax + 1] <- 0
            Shat_temp[1, r.ID %in% jall[ql0mmax]] <- 0
          } else {
            ShatR_b[1, 2:(nj + 1)] <- 0
            Shat_temp[1, r.ID %in% jall] <- 0
          }
        }
        m <- cumprod(ShatR_temp[1, 1:nj])
        for (j in 1:nj){
          Shat_temp[1, r.ID == jall[j]] <- Shat_temp[1, r.ID == jall[j]] * m[j]
        }
      }
      Shat <- Shat_temp[1, -match(TestT, tpntLod)]
    } else {
      if (class(pred[[1]]) == "numeric"){
        Shat <- pred[[Ni]][1:tlen]
      } else {
        ## Compute the estimated survival probability of the Ni-th subject
        Shat_temp <- matrix(0, nrow = 1, ncol = tlen)
        
        r.ID <- findInterval(tpntLod, TestT)
        r.ID[r.ID > TestTIntN] <- TestTIntN
        
        jall <- unique(r.ID[r.ID > 0])
        nj <- length(jall)
        
        ## Deal with left-truncation
        Shat_temp[1, r.ID == 0] <- 1
        
        if(nj == 1){
          ## Get the index of the Pred to compute Shat
          II = which(data$ID == id_uniq[Ni])[jall[nj]]
          Shat_i = ipred::getsurv(pred[[II]], tpntLod[r.ID == jall[nj]])
          Shat_temp[1, r.ID == jall[nj]] <- Shat_i / Shat_i[1]
        } else {
          ShatR_temp <- matrix(0, nrow = 1, ncol = nj + 1)
          ShatR_temp[1, 1] = 1
          
          # S_1(L_1), S_2(L_2), S_3(L_3), ..., S_{nj}(L_{nj})
          qL = rep(0, nj)
          for (j in 1:nj){
            ## Get the index of the Pred to compute Shat
            II <- which(data$ID == id_uniq[Ni])[1] + jall[j] - 1
            Shat_j = ipred::getsurv(pred[[II]], tpntLod[r.ID == jall[j]])
            
            qL[j] <- Shat_j[1]
            # S_{j}(R_{j}), j=1,...nj-1
            jR = ipred::getsurv(pred[[II]], TestT[j + 1])
            ShatR_temp[1, j + 1] = jR / qL[j]
            Shat_temp[1, r.ID == jall[j]] <- Shat_j / qL[j]
          }
          
          ql0 <- which(qL == 0)
          if (length(ql0) > 0){
            if (any(qL > 0)){
              maxqlnot0 <- max(which(qL > 0))
              
              ql0lmax <- ql0[ql0 < maxqlnot0]
              ql0mmax <- ql0[ql0 >= maxqlnot0]
              ShatR_temp[1, ql0lmax + 1] <- 1
              Shat_temp[1, r.ID %in% jall[ql0lmax]] <- 1
              ShatR_temp[1, ql0mmax + 1] <- 0
              Shat_temp[1, r.ID %in% jall[ql0mmax]] <- 0
            } else {
              ShatR_b[1, 2:(nj + 1)] <- 0
              Shat_temp[1, r.ID %in% jall] <- 0
            }
          }
          m <- cumprod(ShatR_temp[1, 1:nj])
          for (j in 1:nj){
            Shat_temp[1, r.ID == jall[j]] <- Shat_temp[1, r.ID == jall[j]] * m[j]
          }
        }
        Shat <- Shat_temp[1, -match(TestT, tpntLod)]
      }
    }
    
  } else {
    pred <- predict(obj.roc, TestData)$pred
    Shat <- getSurv(pred, tpnt)
  }
  
  Shat
}


shat <- function(data, pred = NULL, tpnt = NULL, obj.roc = NULL){
  if (is.null(tpnt)){
    tpnt = c(0, sort(unique(data$Stop)))
  }
  N = length(unique(data$ID))
  Shatt = sapply(1:N, function(Ni) shat_funct(Ni = Ni, data = data, pred = pred, tpnt = tpnt, obj.roc = obj.roc))
  return(Shatt)
}

############################################################################################################
############################################################################################################

# Set the file path for the dataset
file_path <- ""

# Define the file path where you want to save the Parquet file
file_path_save <- ""

# Read the Parquet file into a dataframe
df <- read_parquet(file_path)

# Clean column names by removing "%" and "-"
names(df) <- gsub("%", "", names(df))
names(df) <- gsub("-", "", names(df))

# Convert the dataframe to ensure it's in the correct format
df <- as.data.frame(df)

# Subset the dataframe to include only rows from 6000 to 8000
df <- df[0:4000, ]

# Create a unique identifier for each row based on 'Chunk ID' and 'Customer ID'
df$unique_id <- paste(df$`Chunk ID`, df$`Customer ID`, sep = "_")
df$unique_id <- as.integer(as.factor(df$unique_id))
df$ID <- df$unique_id

# Identify all IDs where every 'event' value is 0
ids_with_all_zero_events <- df %>%
  group_by(ID) %>%
  filter(all(event == 0)) %>%
  ungroup() %>%
  select(ID) %>%
  distinct()

# Filter the original dataframe to include only those IDs
pred_df <- df %>%
  semi_join(ids_with_all_zero_events, by = "ID")

# Create a copy of df for further processing
df2 <- df

# Select and rename relevant columns for analysis
df <- df[, c("Customer ID", "ID", "tstart", "tstop", "event", "Deposit Amount", "euribor_6m", "AEX", "CPI", "Unemployment", "Housing", "GDP", "Knab_rate", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6", "Month_7", "Month_8", "Month_9", "Month_10", "Month_11", "Month_12")]
names(df)[names(df) == "tstop"] <- "Stop"
names(df)[names(df) == "tstart"] <- "Start"
names(df)[names(df) == "event"] <- "Event"
df <- arrange(df, ID)

# Create the survival model formula
Formula = Surv(Start, Stop, Event, type = "counting") ~ euribor_6m + AEX + CPI + Unemployment + Housing + GDP + Knab_rate + Month_2 + Month_3 + Month_4 + Month_5 + Month_6 + Month_7 + Month_8 + Month_9 + Month_10 + Month_11 + Month_12

# Fit a Cox proportional hazards model
Coxobj = coxph(formula = Formula, data = df)

# Convert 'Deposit Amount' to numeric in df2 and prepare deposit amounts data
df2$`Deposit Amount` <- as.numeric(df2$`Deposit Amount`)
deposit_amounts <- df2 %>%
  select(ID, `Deposit Amount`) %>%
  distinct() %>%
  arrange(ID)

# Prepare pred_df for processing
pred_df$unique_id <- paste(pred_df$`Chunk ID`, pred_df$`Customer ID`, sep = "_")
pred_df$unique_id <- as.integer(as.factor(pred_df$unique_id))
pred_df$ID <- pred_df$unique_id
pred_df2 <- pred_df
pred_df <- pred_df[, c("Customer ID", "ID", "tstart", "tstop", "event", "euribor_6m", "AEX", "CPI", "Unemployment", "Housing", "GDP", "Knab_rate", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6", "Month_7", "Month_8", "Month_9", "Month_10", "Month_11", "Month_12")]
names(pred_df)[names(pred_df) == "tstop"] <- "Stop"
names(pred_df)[names(pred_df) == "tstart"] <- "Start"
names(pred_df)[names(pred_df) == "event"] <- "Event"
pred_df <- arrange(pred_df, ID)
pred_df2$`Deposit Amount` <- as.numeric(pred_df2$`Deposit Amount`)
deposit_amounts <- pred_df2 %>%
  select(ID, `Deposit Amount`) %>%
  distinct()
pred_deposit_amounts <- arrange(deposit_amounts, ID)
pred_deposit_amounts <- arrange(pred_deposit_amounts, ID)

# Process prediction in chunks
tpnt <- seq(0, 1826)
chunk_size <- 30
unique_ids <- pred_df %>% distinct(ID) %>% pull(ID)
num_chunks <- ceiling(length(unique_ids) / chunk_size)

# Initialize variables to store aggregated results
all_survival_probs <- list()

# Calculate predicted survival curve
process_chunk_survival_analysis <- function(chunk_index, unique_ids, df, chunk_size, Coxobj, tpnt) {
  start_index <- (chunk_index - 1) * chunk_size + 1
  end_index <- min(chunk_index * chunk_size, length(unique_ids))
  current_ids <- unique_ids[start_index:end_index]
  
  chunk_df <- pred_df %>% filter(ID %in% current_ids)
  chunk_df <- chunk_df[, c("ID", "Start", "Stop", "Event", "euribor_6m", "AEX", "CPI", "Unemployment", "Housing", "GDP", "Knab_rate", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6", "Month_7", "Month_8", "Month_9", "Month_10", "Month_11", "Month_12")]
  
  Pred <- survfit(Coxobj, newdata = chunk_df)
  Shat <- shat(data = chunk_df, pred = Pred, tpnt=tpnt)
  
  return(Shat)
}

# Parallel processing
num_cores <- 7
args_list <- lapply(1:num_chunks, function(i) list(chunk_index = i, unique_ids = unique_ids, df = pred_df, chunk_size = chunk_size, Coxobj = Coxobj, tpnt = tpnt))
all_survival_probs <- mclapply(args_list, function(args) do.call(process_chunk_survival_analysis, args), mc.cores = num_cores)
final_survival_probs <- do.call(cbind, all_survival_probs)
predTnew <- list(survival.times = tpnt, survival.probs = final_survival_probs)

# Extend the survival probabilities and times for forecasting
last_stop_df <- pred_df %>%
  group_by(ID) %>%
  summarise(LastStop = max(Stop)) %>%
  ungroup()

# Extend the timeframe of predictions
max_days <- max(predTnew$survival.times) + 365*5
extended_times <- seq(from = max(predTnew$survival.times), to = max_days, by = 1)

extended_probs <- matrix(NA, nrow = length(extended_times), ncol = ncol(predTnew$survival.probs))
for(i in 1:ncol(extended_probs)) {
  last_prob <- tail(predTnew$survival.probs[,i][!is.na(predTnew$survival.probs[,i])], 1)
  extended_probs[,i] <- rep(last_prob, length(extended_times))
}

predTnew$survival.probs <- rbind(predTnew$survival.probs, extended_probs)
predTnew$survival.times <- c(predTnew$survival.times, extended_times)

# Forecast future probabilities and weight them by deposit amount
calculate_forecast <- function(ID, delta_ts, last_stop_df, predTnew) {
  last_t <- last_stop_df[last_stop_df$ID == ID, "LastStop"]
  last_t <- as.data.frame(last_t)
  last_t <- last_t$LastStop[1]
  forecast_probs <- numeric(length(delta_ts))
  
  for(i in seq_along(delta_ts)) {
    delta_t <- delta_ts[i]
    t_plus_delta <- last_t + delta_t

    closest_time_index <- max(which(predTnew$survival.times <= t_plus_delta))
    closest_time_prob <- predTnew$survival.probs[closest_time_index, ID]

    last_t_index <- max(which(predTnew$survival.times <= last_t))
    last_t_prob <- predTnew$survival.probs[last_t_index, ID]

    forecast_probs[i] <- closest_time_prob / last_t_prob
  }
  
  return(forecast_probs)
}

delta_ts <- seq(0, 1826, by = 1)

weighted_forecast_sum <- numeric(length(delta_ts))
for (ID in unique(pred_deposit_amounts$ID)) {
  forecast_probs <- calculate_forecast(ID, delta_ts, last_stop_df, predTnew)
  deposit_amount <- pred_deposit_amounts[pred_deposit_amounts$ID == ID, "Deposit Amount"]
  if (is.data.frame(deposit_amount) || length(deposit_amount) > 1) {
    deposit_amount <- deposit_amount[1, 1]
  }
  weighted_forecast_sum <- weighted_forecast_sum + (forecast_probs * deposit_amount)
}

weighted_forecast_sum <- weighted_forecast_sum / sum(pred_deposit_amounts$`Deposit Amount`)

# Recreate the dataframe to avoid any confusion with function names
forecast_data <- data.frame(
  Days <- seq(0, 1826, by = 1),
  WeightedForecastSum = weighted_forecast_sum
)

# Create the dataframe using data.frame
results_df <- data.frame(
  ActualTimePoints = delta_ts,
  AverageSurvProbs = weighted_forecast_sum
)

# Save results_df to an Excel file
write.xlsx(results_df, file = file_path_save)

# Print Cox diagnostics
summary(Coxobj)
cox_summary <- summary(Coxobj)
coef_table <- cox_summary$coefficients
coef_df <- as.data.frame(coef_table)
latex_table <- xtable(coef_df)
print.xtable(latex_table, type = "latex")

