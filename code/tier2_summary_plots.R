library(ggplot2)
library(ggrepel)


df_counts <- data.frame(
  category = c("Sample level", "Human diversity", "Donor level"),
  n = c(3, 19, 28)
)

hgca_cols <- c(
  "Sample level" = "#7ab66f",
  "Human diversity" = "#025773",
  "Donor level" = "#c6093b"
)

hgca_navy <- "#002f47"
hgca_grey <- "#b4b4b4"


df_counts <- data.frame(
  category = c("Sample level", "Human diversity", "Donor level"),
  n = c(3, 19, 28)
)

df_counts$total <- sum(df_counts$n)

p_stacked <- ggplot(df_counts, aes(x = "Tier 2 protected", y = n, fill = category)) +
  geom_col(width = 0.6) +
  coord_flip() +
  geom_text(
    aes(label = n),
    position = position_stack(vjust = 0.5),
    size = 4,
    color = "white"
  ) +
  scale_fill_manual(values = hgca_cols) +
  labs(
    x = NULL,
    y = NULL,
    fill = NULL,
    title = "Tier 2 protected metadata field counts"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title.position = "plot"
  )

p_stacked

ggsave( "~/Desktop/tier2summary_stacked_bar.png", p_stacked)

library(ggplot2)

df_counts <- data.frame(
  category = c("Sample level", "Human diversity", "Donor level"),
  n = c(3, 19, 28)
)

df_counts$total <- sum(df_counts$n)

df_tiles <- data.frame(
  id = seq_len(df_counts$total),
  category = rep(df_counts$category, times = df_counts$n)
)

ncol_grid <- 10
df_tiles$x <- ((df_tiles$id - 1) %% ncol_grid) + 1
df_tiles$y <- ((df_tiles$id - 1) %/% ncol_grid) + 1

p_waffle <- ggplot(df_tiles, aes(x = x, y = y, fill = category)) +
  geom_tile(color = "white", linewidth = 0.6) +
  coord_equal() +
  scale_y_reverse(breaks = NULL) +
  scale_x_continuous(breaks = NULL) +
  labs(
    x = NULL,
    y = NULL,
    fill = NULL,
    title = "Schematized protected 'Tier 2' metadata (50 fields)"
  ) +
  scale_fill_manual(values = hgca_cols) +
  theme_minimal(base_size = 16) +
  theme(
    panel.grid = element_blank(),
    axis.text = element_blank(),
    plot.title.position = "plot"
  )

p_waffle


ggsave( "~/Desktop/tier2summary_waffle.png", p_waffle)


hgca_navy <- "#002f47"

df_counts <- data.frame(
  category = c("Sample level", "Human diversity", "Donor level"),
  n = c(3, 19, 28)
)

total_n <- sum(df_counts$n)

df_tiles <- data.frame(
  id = seq_len(total_n),
  category = rep(df_counts$category, times = df_counts$n)
)

ncol_grid <- 10
df_tiles$x <- ((df_tiles$id - 1) %% ncol_grid) + 1
df_tiles$y <- ((df_tiles$id - 1) %/% ncol_grid) + 1

anchors <- rbind(
  transform(subset(df_tiles, category == "Donor level")[8, ], label = "clinical_activity_score"),
  transform(subset(df_tiles, category == "Donor level")[26, ], label = "comorbidity_ontology_term"),
  transform(subset(df_tiles, category == "Human diversity")[5, ], label = "language_mother_father_tongue"),
  transform(subset(df_tiles, category == "Sample level")[2, ], label = "procedure")
)

p_waffle_a <- ggplot(df_tiles, aes(x = x, y = y, fill = category)) +
  geom_tile(color = "white", linewidth = 0.7) +
  geom_point(
    data = anchors,
    aes(x = x, y = y),
    inherit.aes = FALSE,
    size = 1.6,
    color = "black"
  ) +
  geom_text_repel(
    data = anchors,
    aes(x = x, y = y, label = label),
    color = "black",
    inherit.aes = FALSE,
    seed = 10,
    box.padding = 0.4,
    point.padding = 0.35,
    label.padding = unit(0.15, "lines"),
    label.size = 0.25,
    segment.size = 0.35,
    min.segment.length = 0,
    force = 2,
    max.overlaps = Inf,
    direction = "y",
    nudge_x = 3.5
  ) +
  scale_fill_manual(values = hgca_cols) +
  scale_color_manual(values = hgca_cols) +
  coord_equal(clip = "off") +
  scale_y_reverse(breaks = NULL) +
  scale_x_continuous(breaks = NULL, limits = c(0.5, ncol_grid + 5.5)) +
  labs(
    x = NULL,
    y = NULL,
    fill = NULL,
    title = "Tier 2 protected metadata schema (50 fields)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text = element_blank(),
    plot.title.position = "plot",
    plot.title = element_text(color = hgca_navy, face = "bold"),
    legend.text = element_text(color = hgca_navy),
    legend.position = "bottom",
    plot.margin = margin(10, 140, 10, 10)
  ) +
  guides(color = "none")

p_waffle_a

ggsave( "~/Desktop/tier2summary_waffle.svg", p_waffle_a)
