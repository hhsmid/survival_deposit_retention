# Load necessary libraries
library(LTRCforests)
library(survival)
library(arrow)
library(dplyr)
library(openxlsx)
library(magrittr)
library(data.table)
library(ggplot2)
library(parallel)

# Define the file path to the dataset
file_path <- ""

# Define the file path for saving the results in an Excel file
file_path_save <- ""

# Read the parquet file into a dataframe
df <- read_parquet(file_path)

# Clean column names by removing "%" and "-" characters
names(df) <- gsub("%", "", names(df))
names(df) <- gsub("-", "", names(df))

# Convert the dataframe to a data.frame object for compatibility
df <- as.data.frame(df)

# Subset the dataframe to include only rows from 6000 to 8000
#df <- df[1000:4000, ]

# Create a unique chunk identifier
df$unique_id <- paste(df$`Chunk ID`, df$`Customer ID`, sep = "_")
df$unique_id <- as.integer(as.factor(df$unique_id))
df$ID <- df$unique_id

# Identify IDs where all events are zero
ids_with_all_zero_events <- df %>%
  group_by(ID) %>%
  filter(all(event == 0)) %>%
  ungroup() %>%
  select(ID) %>%
  distinct()

pred_df <- df %>%
  semi_join(ids_with_all_zero_events, by = "ID")

# Duplicate the original dataframe for backup purposes
df2 <- df

# Select and reorder specific columns in the dataframe
df <- df[, c("ID", "tstart", "tstop", "event", "Deposit Amount", "euribor_6m", "AEX", 
             "CPI", "Unemployment", "Housing", "GDP", "Knab_rate", "Month_2", "Month_3", 
             "Month_4", "Month_5", "Month_6", "Month_7", "Month_8", "Month_9", 
             "Month_10", "Month_11", "Month_12")]

# Rename columns for clarity
names(df)[names(df) == "tstop"] <- "Stop"
names(df)[names(df) == "tstart"] <- "Start"
names(df)[names(df) == "event"] <- "Event"

# Define survival analysis formula
Formula = Surv(`Start`, `Stop`, Event, type = "counting") ~ euribor_6m + AEX + CPI + 
  Unemployment + Housing + GDP + Knab_rate + Month_2 + Month_3 + Month_4 + 
  Month_5 + Month_6 + Month_7 + Month_8 + Month_9 + Month_10 + Month_11 + 
  Month_12

# Fit an LTRCCIF model to the data
LTRCCIFobj = ltrccif(formula = Formula, data = df, id = ID, ntree = 10L, mtry = 5, cores = 7)

# Define a sequence of time points for the plot
tpnt <- seq(0, 1826)
newData <- df

# Create a unique identifier in the prediction dataframe
pred_df$unique_id <- paste(pred_df$`Chunk ID`, pred_df$`Customer ID`, sep = "_")
pred_df$unique_id <- as.integer(as.factor(pred_df$unique_id))
pred_df$ID <- pred_df$unique_id

# Backup the modified prediction dataframe
pred_df2 <- pred_df

# Select and reorder specific columns in the prediction dataframe
pred_df <- pred_df[, c("ID", "tstart", "tstop", "event", "euribor_6m", "AEX", "CPI", 
                       "Unemployment", "Housing", "GDP", "Knab_rate", "Month_2", 
                       "Month_3", "Month_4", "Month_5", "Month_6", "Month_7", 
                       "Month_8", "Month_9", "Month_10", "Month_11", "Month_12")]

# Rename columns for clarity in the prediction dataframe
names(pred_df)[names(pred_df) == "tstop"] <- "Stop"
names(pred_df)[names(pred_df) == "tstart"] <- "Start"
names(pred_df)[names(pred_df) == "event"] <- "Event"

# Arrange the prediction dataframe by ID
pred_df <- arrange(pred_df, ID)

# Convert `Deposit Amount` to numeric for analysis
pred_df2$`Deposit Amount` <- as.numeric(pred_df2$`Deposit Amount`)

# Select distinct `Deposit Amount` values associated with each ID
deposit_amounts <- pred_df2 %>%
  select(ID, `Deposit Amount`) %>%
  distinct()

# Arrange the `Deposit Amount` data by ID for prediction
pred_deposit_amounts <- arrange(deposit_amounts, ID)
pred_deposit_amounts <- arrange(pred_deposit_amounts, ID)

# Define the chunk processing function for survival analysis
process_chunk_survival_analysis <- function(chunk_index, unique_ids, pred_df, chunk_size, LTRCCIFobj, tpnt) {
  start_index <- (chunk_index - 1) * chunk_size + 1
  end_index <- min(chunk_index * chunk_size, length(unique_ids))
  current_ids <- unique_ids[start_index:end_index]
  
  # Filter the prediction dataframe for the current chunk of IDs
  chunk_df <- pred_df %>% filter(ID %in% current_ids)
  
  # Predict survival probabilities for the chunk
  Pred <- predictProb(object = LTRCCIFobj, newdata = chunk_df, time.eval = tpnt, newdata.id = ID)
  
  # Return the survival probabilities
  return(Pred$survival.probs)
}

# Prepare for parallel execution
num_cores <- 7
unique_ids <- unique(pred_df$ID)
chunk_size <- 30
num_chunks <- ceiling(length(unique_ids) / chunk_size)

# Create a list of arguments for each chunk
args_list <- lapply(1:num_chunks, function(i) list(chunk_index = i, unique_ids = unique_ids, pred_df = pred_df, chunk_size = chunk_size, LTRCCIFobj = LTRCCIFobj, tpnt = tpnt))

# Execute the chunk processing in parallel
all_survival_probs <- mclapply(args_list, function(args) do.call(process_chunk_survival_analysis, args), mc.cores = num_cores)

# Combine the results. Assuming survival probabilities need to be column-bound since each represents an ID's survival probability over time
final_survival_probs <- do.call(cbind, all_survival_probs)

# Wrap the final results in a list with survival times for completeness
predTnew <- list(survival.times = tpnt, survival.probs = final_survival_probs)

# Create a dataframe to track the maximum 'Stop' time for each ID
last_stop_df <- pred_df %>%
  group_by(ID) %>%
  summarise(LastStop = max(Stop)) %>%
  ungroup()

# Determine the maximum number of days to extend survival probabilities
max_days <- max(predTnew$survival.times) + 365*5 # Extend by 5 years

# Generate a sequence of time points from the last time in predTnew to max_days
extended_times <- seq(from = max(predTnew$survival.times), to = max_days, by = 1)

# Initialize a matrix for extended survival probabilities with NA values
extended_probs <- matrix(NA, nrow = length(extended_times), ncol = ncol(predTnew$survival.probs))

# Loop through each column (ID) to extend survival probabilities
for(i in 1:ncol(extended_probs)) {
  last_prob <- tail(predTnew$survival.probs[,i][!is.na(predTnew$survival.probs[,i])], 1)
  extended_probs[,i] <- rep(last_prob, length(extended_times))
}

# Combine the original and extended survival probabilities
full_survival_probs <- rbind(predTnew$survival.probs, extended_probs)

# Combine the original survival times with the newly extended times
full_times <- c(predTnew$survival.times, extended_times)

# Update predTnew with the extended survival probabilities and times
predTnew$survival.probs <- full_survival_probs
predTnew$survival.times <- full_times

# Define a function to calculate forecast probabilities for a given ID over a series of time deltas
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

# Define a sequence of days from 0 to 1826 (5 years), stepping by 1 day at a time
delta_ts <- seq(0, 1826, by = 1)

# Calculate forecast probabilities for ID 9 over the specified time deltas
forecast_probs_example <- calculate_forecast(ID = 9, delta_ts, last_stop_df, predTnew)

# Initialize a numeric vector for the weighted forecast sum
weighted_forecast_sum <- numeric(length(delta_ts))

# Loop through each unique ID in pred_deposit_amounts
for (ID in unique(pred_deposit_amounts$ID)) {
  forecast_probs <- calculate_forecast(ID, delta_ts, last_stop_df, predTnew)
  
  deposit_amount <- pred_deposit_amounts[pred_deposit_amounts$ID == ID, "Deposit Amount"]
  if (is.data.frame(deposit_amount) || length(deposit_amount) > 1) {
    deposit_amount <- deposit_amount[1, 1]  # Extract single deposit amount if needed
  }
  weighted_forecast_sum <- weighted_forecast_sum + (forecast_probs * deposit_amount)
}

# Calculate the total deposit amount for normalization
total_deposit_amount <- sum(pred_deposit_amounts$`Deposit Amount`)

# Normalize the weighted forecast sum by the total deposit amount
weighted_forecast_sum <- weighted_forecast_sum / total_deposit_amount

# Initialize a dataframe for storing the days and the corresponding weighted forecast sum
forecast_data <- data.frame(
  Days = seq(0, 1826, by = 1),
  WeightedForecastSum = weighted_forecast_sum
)

# Create a dataframe for actual time points and average survival probabilities
results_df <- data.frame(
  ActualTimePoints = delta_ts,
  AverageSurvProbs = weighted_forecast_sum
)

# Save the results dataframe to an Excel file
write.xlsx(results_df, file = file_path_save)

# Print the LTRCCIF model object for review
print(LTRCCIFobj)

