library(ggplot2)
library(ggthemes)
library(tidyverse)
library(rstatix)
library(ARTool)
library(emmeans)
library(ggsignif)
library(reshape2)
library(gridExtra)
library(colorBlindness)
library(statsr)
library(ggrain)
library(see)
library(ggpp)
library(ggrepel)

autowebglm_df <- read.csv('data/AutoWebGLM.csv', sep=",")
MindAct_df <- read.csv('data/mindact_final.csv', sep=",")
MultiUI <- read.csv('data/MultiUI.csv', sep=",")
Synapse_df <- read.csv('data/Synapse.csv', sep=",")
Synatra_df <- read.csv('data/Synatra.csv', sep=",")

autowebglm_df$Time
convert_time_to_seconds <- function(time_str) {
  parts <- as.numeric(strsplit(time_str, ":")[[1]])
  return(parts[1] * 3600 + parts[2] * 60 + parts[3])
}

autowebglm_df$Time_seconds <- sapply(autowebglm_df$Time, convert_time_to_seconds)
MindAct_df$Time_seconds <- sapply(MindAct_df$Time, convert_time_to_seconds)
MultiUI$Time_seconds <- sapply(MultiUI$Time, convert_time_to_seconds)
Synapse_df$Time_seconds <- sapply(Synapse_df$Time, convert_time_to_seconds)
Synatra_df$Time_seconds <- sapply(Synatra_df$Time, convert_time_to_seconds)

autowebglm_df$Time_min <- autowebglm_df$Time_seconds/60
MindAct_df$Time_min <- MindAct_df$Time_seconds/60
MultiUI$Time_min <- MultiUI$Time_seconds/60
Synapse_df$Time_min <- Synapse_df$Time_seconds/60
Synatra_df$Time_min <- Synatra_df$Time_seconds/60

autowebglm_df

autowebglm_df <- autowebglm_df %>%
  select(Run, GPU, Split, Time_min)
MindAct_df <- MindAct_df %>%
  select(Run, GPU, Split, Time_min)
MultiUI <- MultiUI %>%
  select(Run, GPU, Split, Time_min)
Synapse_df <- Synapse_df %>%
  select(Run, GPU, Split, Time_min)
Synatra_df <- Synatra_df %>%
  select(Run, GPU, Split, Time_min)

autowebglm_df <- autowebglm_df %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Time = sum(Time_min),
    .groups = 'drop'
  )

MindAct_df <- MindAct_df %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Time = sum(Time_min),
    .groups = 'drop'
  )

MultiUI <- MultiUI %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Time = sum(Time_min),
    .groups = 'drop'
  )

Synapse_df <- Synapse_df %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Time = sum(Time_min),
    .groups = 'drop'
  )

Synatra_df <- Synatra_df %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Time = sum(Time_min),
    .groups = 'drop'
  )


MindAct_df["agent"] = "saMindAct"
autowebglm_df["agent"] = "AutoWebGLM"
MultiUI["agent"] = "MultiUI"
Synapse_df["agent"] = "Synapse"
Synatra_df["agent"] = "Synatra"

combined_df <- rbind(MindAct_df, autowebglm_df, MultiUI, Synapse_df, Synatra_df)
combined_df
combined_df <- combined_df[combined_df$GPU != "V100-32GB", ]
df3<- combined_df %>% group_by(agent, GPU) %>% get_summary_stats(Total_Time)
df3
print(df3, n=100)
