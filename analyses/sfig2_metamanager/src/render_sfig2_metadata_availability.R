#!/usr/bin/env Rscript
## Supplementary Figure 2 — metadata-availability donut.
## Plot script from the paper tree; schema validation lives in MetaManager
## (https://github.com/CellDiscoveryNetwork/MetaManager, commit 9bed5fc73ef2).

args <- commandArgs(trailingOnly = TRUE)
here <- if (length(grep("^--file=", commandArgs(FALSE)))) {
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
} else getwd()
repo <- normalizePath(file.path(here, "..", "..", ".."))
in_path <- if (length(args) >= 1) {
  args[[1]]
} else {
  file.path(repo, "data", "sfig2", "MetadataAvailability-data.txt")
}
out_base <- if (length(args) >= 2) {
  args[[2]]
} else {
  file.path(dirname(here), "out", "sfig2_metadata_availability")
}
dir.create(dirname(out_base), recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggforce)
  library(ggnewscale)
})

df <- read.delim(in_path, sep = "\t", stringsAsFactors = FALSE)
df <- df %>%
  mutate(
    SubCategory = if_else(
      is.na(SubCategory) | SubCategory == "NA",
      Title,
      SubCategory
    )
  )

PD <- df %>%
  group_by(Title, SubCategory) %>%
  summarise(n = sum(Count), .groups = "drop") %>%
  mutate(
    SubCategory_Unique = if_else(
      SubCategory == "Other", paste0(SubCategory, "_", Title), SubCategory
    )
  )

total_counts <- sum(PD$n)
inner_data <- PD %>%
  group_by(Title) %>%
  summarise(value = sum(n), .groups = "drop") %>%
  mutate(
    fraction = value / total_counts,
    ymax = cumsum(fraction) * 2 * pi,
    ymin = lag(ymax, default = 0),
    mid_angle = (ymin + ymax) / 2,
    label_text = as.character(value),
    legend_text = paste0(Title, " (", round(fraction * 100, 1), "%)"),
    x_center = 0.525 * sin(mid_angle),
    y_center = 0.525 * cos(mid_angle)
  )

outer_data <- PD %>%
  mutate(
    fraction = n / total_counts,
    ymax = cumsum(fraction) * 2 * pi,
    ymin = lag(ymax, default = 0),
    mid_angle = (ymin + ymax) / 2,
    label_text = as.character(n),
    legend_text = paste0(SubCategory, " (", round(fraction * 100, 1), "%)"),
    x_center = 0.815 * sin(mid_angle),
    y_center = 0.815 * cos(mid_angle)
  )

parent_hues <- c(
  Both = "#444444",
  Blue = "#0072b2",
  Violet = "#cc79a7",
  Amber = "#e69f00"
)
unique_titles <- unique(PD$Title)
inner_palette <- c()
for (i in seq_along(unique_titles)) {
  t <- unique_titles[i]
  if (t == "Both") {
    inner_palette[t] <- parent_hues["Both"]
  } else {
    root_idx <- ((i - 1) %% (length(parent_hues) - 1)) + 2
    inner_palette[t] <- parent_hues[root_idx]
  }
}
inner_labels <- setNames(inner_data$legend_text, inner_data$Title)

outer_palette <- c()
for (t in unique_titles) {
  subs_for_title <- PD %>%
    filter(Title == t) %>%
    pull(SubCategory_Unique) %>%
    unique()
  n_subs <- length(subs_for_title)
  base_hex <- inner_palette[t]
  if (t == "Both") {
    shades <- colorRampPalette(c("#999999", "#222222"))(n_subs)
  } else if (base_hex == "#0072b2") {
    shades <- colorRampPalette(c("#56b4e9", "#004b75"))(n_subs)
  } else if (base_hex == "#cc79a7") {
    shades <- colorRampPalette(c("#f3d5e5", "#782353"))(n_subs)
  } else {
    shades <- colorRampPalette(c("#f0e442", "#a67300"))(n_subs)
  }
  names(shades) <- subs_for_title
  for (s in subs_for_title) {
    if (grepl("^Other_", s)) shades[s] <- "#009e73"
  }
  outer_palette <- c(outer_palette, shades)
}

outer_labels_map <- setNames(outer_data$legend_text, outer_data$SubCategory_Unique)
outer_labels_map[grep("^Other_", names(outer_labels_map))] <- "Other"
outer_breaks <- names(outer_palette)
other_items <- outer_breaks[grep("^Other_", outer_breaks)]
if (length(other_items) > 1) {
  outer_breaks <- outer_breaks[!outer_breaks %in% other_items[-1]]
}
outer_labels_final <- outer_labels_map[outer_breaks]

p <- ggplot() +
  geom_arc_bar(
    data = inner_data,
    aes(x0 = 0, y0 = 0, r0 = 0.4, r = 0.65, start = ymin, end = ymax, fill = Title),
    color = "white", linewidth = 0.5
  ) +
  geom_text(
    data = inner_data,
    aes(x = x_center, y = y_center, label = label_text),
    family = "Helvetica", color = "black", fontface = "bold", size = 2.4
  ) +
  scale_fill_manual(values = inner_palette, labels = inner_labels, name = "Metadata Category") +
  new_scale_fill() +
  geom_arc_bar(
    data = outer_data,
    aes(x0 = 0, y0 = 0, r0 = 0.68, r = 0.95, start = ymin, end = ymax, fill = SubCategory_Unique),
    color = "white", linewidth = 0.3
  ) +
  geom_text(
    data = outer_data,
    aes(x = x_center, y = y_center, label = label_text),
    family = "Helvetica", color = "black", fontface = "bold", size = 2.4
  ) +
  scale_fill_manual(
    values = outer_palette,
    labels = outer_labels_final,
    breaks = outer_breaks,
    name = "Specific Variable"
  ) +
  coord_fixed(xlim = c(-1.1, 1.1), ylim = c(-1.1, 1.1)) +
  theme_void() +
  theme(
    text = element_text(family = "Helvetica", color = "black"),
    legend.title = element_text(face = "bold", size = 8),
    legend.text = element_text(size = 7)
  )

ggsave(paste0(out_base, ".pdf"), p, width = 6.2, height = 4.5, units = "in")
ggsave(paste0(out_base, ".png"), p, width = 6.2, height = 4.5, units = "in", dpi = 300, bg = "white")
message("Wrote ", out_base, ".{pdf,png}")
