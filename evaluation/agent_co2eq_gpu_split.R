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

autowebglm_df


convert_time_to_seconds <- function(time_str) {
  parts <- as.numeric(strsplit(time_str, ":")[[1]])
  return(parts[1] * 3600 + parts[2] * 60 + parts[3])
}

MindAct_df$Time_seconds <- sapply(MindAct_df$Time, convert_time_to_seconds)

# Filter for relevant components and sum up
summed_data <- MindAct_df %>%
  filter(Component %in% c("action_pred", "cand_gen")) %>%
  group_by(Run, GPU, Split) %>%
  summarise(
    Energy = sum(Energy),
    CO2eq = sum(CO2eq),
    Sum_Time_seconds = sum(Time_seconds), # Sum in seconds
    Sum_Distance = sum(Distance),
    .groups = 'drop'
  )

# If you want to convert Sum_Time_seconds back to HH:MM:SS format
convert_seconds_to_time <- function(seconds) {
  h <- floor(seconds / 3600)
  m <- floor((seconds %% 3600) / 60)
  s <- seconds %% 60
  sprintf("%02d:%02d:%02d", h, m, s)
}

summed_data$Sum_Time_HHMMSS <- sapply(summed_data$Sum_Time_seconds, convert_seconds_to_time)

# Print the summed data
MindAct_df <- summed_data %>%
  select(Run, GPU, Split, CO2eq)

MindAct_df <- MindAct_df %>%
  group_by(Run, GPU) %>%
  summarise(
    CO2eq = sum(CO2eq),
    .groups = 'drop'
  )

#autowebglm
autowebglm_df <- autowebglm_df %>%
  select(Run, GPU, Split, CO2eq)

autowebglm_df <- autowebglm_df %>%
  group_by(Run, GPU) %>%
  summarise(
    CO2eq = sum(CO2eq),
    .groups = 'drop'
  )

#MultiUI
MultiUI <- MultiUI %>%
  select(Run, GPU, Split, CO2eq)

MultiUI <- MultiUI %>%
  group_by(Run, GPU) %>%
  summarise(
    CO2eq = sum(CO2eq),
    .groups = 'drop'
  )

#Synapse_df
Synapse_df <- Synapse_df %>%
  select(Run, GPU, Split, CO2eq)

Synapse_df <- Synapse_df %>%
  group_by(Run, GPU) %>%
  summarise(
    CO2eq = sum(CO2eq),
    .groups = 'drop'
  )

#Synatra_df
Synatra_df <- Synatra_df %>%
  select(Run, GPU, Split, CO2eq)

Synatra_df <- Synatra_df %>%
  group_by(Run, GPU) %>%
  summarise(
    CO2eq = sum(CO2eq),
    .groups = 'drop'
  )

MindAct_df["agent"] = "MindAct"
autowebglm_df["agent"] = "AutoWebGLM"
MultiUI["agent"] = "MultiUI"
Synapse_df["agent"] = "Synapse"
Synatra_df["agent"] = "Synatra"
MindAct_df
autowebglm_df

combined_df <- rbind(MindAct_df, autowebglm_df, MultiUI, Synapse_df, Synatra_df)
combined_df
combined_df <- combined_df[combined_df$GPU != "V100-32GB", ] # something went wrong with the V100 and MultiUI until it is fixed remove it for all GPUs

ggplot(combined_df, aes(x = agent, y = CO2eq, color=GPU, fill=Split)) +
  geom_boxplot() +
  labs(
    title = "Energy Consumption by GPU",
    x = "Agent",
    y = "Consumed Energy (kWh)",
    color = "GPU"
  ) +
  theme_minimal() +
  scale_color_okabeito()
# theme(axis.text.x = element_text(angle = 0, hjust = 1)) # Rotate x-axis labels for readability
ggsave("agent_co2eq_gpu_split.pdf", device=cairo_pdf, height=10, width=15, units = "cm")

df <- combined_df[combined_df$GPU=="H100-PCI",]

print(df, n= 300)

df %>% group_by(agent, GPU) %>% get_summary_stats(CO2eq)

df2 <- combined_df[combined_df$GPU=="RTXA6000",]
df2 %>% group_by(agent, GPU) %>% get_summary_stats(CO2eq)


