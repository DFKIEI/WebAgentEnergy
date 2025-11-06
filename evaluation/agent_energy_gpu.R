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

MindAct_df$Time_seconds <- sapply(MindAct_df$Time, convert_time_to_seconds)

# Filter for relevant components and sum up
summed_data <- MindAct_df %>%
  filter(Component %in% c("action_pred", "cand_gen")) %>%
  group_by(Run, GPU, Split) %>%
  summarise(
    Sum_Energy = sum(Energy),
    Sum_CO2eq = sum(CO2eq),
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
summed_data$Sum_Time_seconds

# Print the summed data
print(summed_data)
MindAct_df <- summed_data %>%
  select(Run, GPU, Split, Sum_Energy)

MindAct_df <- MindAct_df %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Energy = sum(Sum_Energy),
    .groups = 'drop'
  )

#autowebglm
autowebglm_df <- autowebglm_df %>%
  select(Run, GPU, Split, Energy)

autowebglm_df <- autowebglm_df %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Energy = sum(Energy),
    .groups = 'drop'
  )
#MultiUI
MultiUI <- MultiUI %>%
  select(Run, GPU, Split, Energy)

MultiUI <- MultiUI %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Energy = sum(Energy),
    .groups = 'drop'
  )
#Synapse_df
Synapse_df <- Synapse_df %>%
  select(Run, GPU, Split, Energy)

Synapse_df <- Synapse_df %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Energy = sum(Energy),
    .groups = 'drop'
  )
#Synatra_df
Synatra_df <- Synatra_df %>%
  select(Run, GPU, Split, Energy)

Synatra_df <- Synatra_df %>%
  group_by(Run, GPU) %>%
  summarise(
    Total_Energy = sum(Energy),
    .groups = 'drop'
  )
MindAct_df["agent"] = "saMindAct"
autowebglm_df["agent"] = "AutoWebGLM"
MultiUI["agent"] = "MultiUI"
Synapse_df["agent"] = "Synapse"
Synatra_df["agent"] = "Synatra"

combined_df <- rbind(MindAct_df, autowebglm_df, MultiUI, Synapse_df, Synatra_df)
combined_df
combined_df <- combined_df[combined_df$GPU != "V100-32GB", ] # something went wrong with the V100 and MultiUI until it is fixed remove it for all Agents

ggplot(combined_df, aes(x = agent, y = Total_Energy, color=GPU)) +
  geom_boxplot() +
  labs(
    title = "Energy Consumption by GPU",
    x = "Agent",
    y = "Consumed Energy (kWh)",
    color = "GPU"
  ) +
  theme_minimal() +
  scale_color_okabeito()+
  scale_x_discrete(labels=c("AutoWebGLM", "MultiUI", "MindAct", "Synapse", "Synatra"))+
  geom_hline(yintercept=-0.5)+
  annotate("text", x = 1, y =-0.8, label = "less")+
  annotate("text", x = 5, y =-0.8, label = "more")+
  annotate("text", x = 3, y =-0.3, label = "energy consumed")

ggplot(combined_df, aes(x = agent, y = Total_Energy, color=GPU)) +
  geom_boxplot() +
  labs(
    x = "Agent",
    y = "Energy Consumption (kWh)",
    color = "GPU"
  ) +
  theme_minimal() +
  theme(legend.position = c(0.15, 0.6), legend.background = element_rect(fill="white", size=0.5, linetype="solid"))+
  scale_color_okabeito(labels=c("A100-40GB", "A100-PCI", "H100", "H100-PCI", "H200", "L40S", "RTX 3090", "RTX A6000")) +
  scale_x_discrete(labels=c("AutoWebGLM", "MultiUI", "MindAct", "Synapse", "Synatra")) +
  geom_segment(aes(x = 1, xend = 5, y = 9.25, yend = 9.25), # Changed from geom_hline to geom_segment
               arrow = arrow(ends = "both", length = unit(0.2, "cm")), # Added arrows to both ends
               color = "black", linewidth = 0.5) + # Set color and size as needed
  annotate("text", x = 1, y = 9, label = "less") +
  annotate("text", x = 5, y = 9, label = "more") +
  annotate("text", x = 3, y = 9.5, label = "Energy Consumption")
ggsave("agent_energy_gpu_v2.pdf", device=cairo_pdf, height=15, width=15, units = "cm")

combined_df$
df <- combined_df[combined_df$GPU=="H100-PCI",]

print(df, n= 300)

df %>% group_by(agent, GPU) %>% get_summary_stats(Total_Energy)

df2 <- combined_df[combined_df$GPU=="RTXA6000",]
df2 %>% group_by(agent, GPU) %>% get_summary_stats(Total_Energy)

df3<- combined_df %>% group_by(agent, GPU) %>% get_summary_stats(Total_Energy)

print(df3, n= 100)
