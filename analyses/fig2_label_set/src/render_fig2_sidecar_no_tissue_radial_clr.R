#!/usr/bin/env Rscript
# Fig. 2 sidecar variant: post-CAP SVG heatmaps WITHOUT tissue-segment or
# radial-layer CLR columns. Collection-method CLR is retained.
#
# Reads existing lineage sidecar CSVs + post-CAP dendrogram SVGs.
# Writes NEW filenames only (does not overwrite the full sidecars).

suppressPackageStartupMessages({
  library(tidyverse)
  library(xml2)
  library(svglite)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Cannot determine script location")
script_path <- normalizePath(sub("^--file=", "", script_arg))
figure_dir <- normalizePath(file.path(dirname(script_path), ".."))
gca_root <- normalizePath(file.path(figure_dir, "..", ".."))
data_dir <- file.path(figure_dir, "data")
out_dir <- file.path(figure_dir, "out")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

drop_groups <- c("Tissue", "Radial layer")
lineage_order <- c("lymphoid", "stromal", "epithelial", "myeloid")
lineage_display <- c(
  lymphoid = "Lymphoid",
  stromal = "Stromal",
  epithelial = "Epithelial",
  myeloid = "Myeloid"
)

reference_leaf_orders <- list(
  lymphoid = c(
    "NK Cells", "ILC3", "GC B Light Zone (GC B LZ)",
    "GC B Dark Zone (GC B DZ)", "Memory B", "Plasma IGG", "Plasma IGA",
    "Naive B", "MAIT Cells", "Gamma Delta T Cells", "NKT Cells",
    "CD8 Memory Exhausted", "CD8 Naive", "CD8 Effector Memory", "CD8 IEL",
    "CD8 Circulating Effector Memory", "CD8 TRM", "CD4 Tr1", "CD4 Tfh",
    "CD4 Naive", "CD4 Tfr", "CD4 tTreg", "CD4 pTreg", "CD4 Th17",
    "CD4 Memory"
  ),
  stromal = c(
    "Adipocytes", "Smooth Muscle Cells (SMC)", "Glia",
    "Secretory Pericytes", "Angiogenic Pericytes", "Contractile Pericytes",
    "Venular Endothelial", "Medullary Sinus Endothelial",
    "Post Arteriole Capillary Endothelial (PAC)", "Arteriolar Endothelial",
    "Interstitial Cells of Cajal (ICC)",
    "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)",
    "Follicular Dendritic Cells (fDC)", "Marginal Reticular Cells (MRC)",
    "Myofibroblasts", "Submucosal Fibroblasts (S3)",
    "Muscularis Propria Fibroblasts", "Crypt Bottom Fibroblasts (S2A)",
    "Crypt Top Fibroblasts (S2B)", "Lamina propria Fibroblasts (S1)"
  ),
  epithelial = c(
    "Microfold Cells (M Cells)", "EEC S", "EEC Progenitors", "EEC L",
    "EEC N", "EEC Enterochromaffin (EC)", "Brunners Gland Cells",
    "Foveolar Cells", "Paneth Cells", "Tuft Progenitors",
    "Mature Goblet Cells", "Intestinal Stem Cells (ISC)",
    "Secretory Progenitors", "BEST4 Enterocytes", "Enterocyte Progenitors",
    "Lower Villus Enterocytes", "Mid Villus Enterocytes",
    "Villus Tip Enterocytes", "Mid Crypt Colonocytes", "BEST4 Colonocytes",
    "Colonocyte Progenitors", "Lower Crypt Colonocytes",
    "Crypt Top Colonocytes"
  ),
  myeloid = c(
    "Nonclassical Monocytes", "Classical Monocytes", "Cycling Macrophages",
    "Follicle Associated Resident Macrophages",
    "Perivascular Resident Macrophages", "M0 Macrophages",
    "Homeostatic Macrophages", "Monocyte Derived Dendritic Cells (MO DC)",
    "pDC", "cDC1", "migDC", "Tolerogenic cDC2", "Neutrophils",
    "Eosinophils", "Mast Cells"
  )
)

postcap_svg_paths <- setNames(
  file.path(
    gca_root, "ARBOL",
    paste0(names(reference_leaf_orders),
           "_annotated_taxonomy_dendrogram_PostCAP_V1.svg")
  ),
  names(reference_leaf_orders)
)

normalize_celltype_label <- function(x) {
  str_squish(str_replace_all(as.character(x), "[\\r\\n]+", " "))
}

svg_number <- function(x) {
  as.numeric(str_extract(as.character(x), "-?[0-9]+(?:\\.[0-9]+)?"))
}

svg_escape <- function(x) {
  x %>%
    str_replace_all("&", "&amp;") %>%
    str_replace_all("<", "&lt;") %>%
    str_replace_all(">", "&gt;") %>%
    str_replace_all('"', "&quot;")
}

postcap_branch_x_scale <- 1.5
postcap_cell_width <- 12
postcap_column_label_size <- 14

read_postcap_svg_tree <- function(slug) {
  doc <- xml2::read_xml(postcap_svg_paths[[slug]])
  root <- xml2::xml_root(doc)
  view_box <- svg_number(str_split(xml2::xml_attr(root, "viewBox"), "\\s+")[[1]])
  width <- view_box[[3]]
  height <- view_box[[4]]

  leaf_text <- xml2::xml_find_all(
    doc,
    ".//*[local-name()='text' and contains(@style, 'font-size: 17.07px')]"
  )
  leaf_positions <- tibble(
    label = xml2::xml_text(leaf_text),
    label_key = normalize_celltype_label(label),
    x = svg_number(xml2::xml_attr(leaf_text, "x")),
    y = svg_number(xml2::xml_attr(leaf_text, "y")),
    text_length = svg_number(xml2::xml_attr(leaf_text, "textLength"))
  )
  scaled_leaf_x <- svg_number(xml2::xml_attr(leaf_text, "x")) *
    postcap_branch_x_scale
  walk2(
    leaf_text, scaled_leaf_x,
    ~ xml2::xml_set_attr(.x, "x", sprintf("%.2f", .y))
  )
  leaf_positions$x <- leaf_positions$x * postcap_branch_x_scale
  reference_keys <- normalize_celltype_label(reference_leaf_orders[[slug]])
  leaf_positions <- leaf_positions %>%
    slice(match(reference_keys, label_key))
  if (any(is.na(leaf_positions$label_key))) {
    stop("Could not recover all leaf coordinates from post-CAP SVG: ", slug)
  }

  polyline_nodes <- xml2::xml_find_all(doc, ".//*[local-name()='polyline']")
  line_nodes <- xml2::xml_find_all(doc, ".//*[local-name()='line']")
  line_nodes <- line_nodes[
    svg_number(xml2::xml_attr(line_nodes, "x1")) <= 130 &
      svg_number(xml2::xml_attr(line_nodes, "x2")) <= 130
  ]
  edge_nodes <- c(as.list(polyline_nodes), as.list(line_nodes))
  walk(edge_nodes, function(node) {
    style <- xml2::xml_attr(node, "style")
    style <- str_replace(style, "stroke-width: [^;]+", "stroke-width: 3.2")
    style <- str_replace(style, "stroke-linecap: [^;]+", "stroke-linecap: round")
    xml2::xml_set_attr(node, "style", style)
    xml2::xml_set_attr(node, "fill", "none")
    xml2::xml_set_attr(node, "stroke-width", "3.2")
    xml2::xml_set_attr(node, "stroke-linecap", "round")
    xml2::xml_set_attr(node, "vector-effect", "non-scaling-stroke")
  })
  edge_markup <- paste(
    c(as.character(polyline_nodes), as.character(line_nodes)),
    collapse = "\n"
  )
  tree_markup <- paste0(
    "<g transform='scale(", postcap_branch_x_scale, " 1)'>",
    edge_markup, "</g>\n",
    paste(as.character(leaf_text), collapse = "\n")
  )
  tree_width <- max(
    width,
    leaf_positions$x + coalesce(leaf_positions$text_length, 0),
    na.rm = TRUE
  ) + 6

  list(
    slug = slug,
    width = width,
    height = height,
    tree_width = tree_width,
    leaf_positions = leaf_positions,
    markup = tree_markup
  )
}

interpolate_svg_color <- function(value, colors, limits) {
  if (is.na(value)) return("#FFFFFF")
  value <- scales::squish(value, range = limits)
  position <- (value - limits[[1]]) / diff(limits)
  palette <- grDevices::colorRamp(colors, space = "Lab")
  rgb <- palette(position)
  grDevices::rgb(rgb[[1]], rgb[[2]], rgb[[3]], maxColorValue = 255)
}

sidecar_tile_color <- function(value, group) {
  if (group == "Atlas") {
    return(interpolate_svg_color(value, c("#F2F2F2", "#111111"), c(0, 1)))
  }
  if (group == "Author hierarchy") {
    return(interpolate_svg_color(value, c("#F2F2F2", "#CC79A7"), c(0, 1)))
  }
  if (group == "PanGI") {
    return(interpolate_svg_color(value, c("#F2F2F2", "#D9782D"), c(0, 1)))
  }
  if (group == "CAP") {
    return(interpolate_svg_color(value, c("#F2F2F2", "#0096A6"), c(0, 1)))
  }
  if (group == "Collection method") {
    return(interpolate_svg_color(
      value, c("#D9D9D9", "#FFFFFF", "#56B4E9"), c(-2.25, 2.25)
    ))
  }
  interpolate_svg_color(value, c("#F2F2F2", "#3A6EA5"), c(0, 1))
}

postcap_svg_defs <- function() {
  paste0(
    "<defs>",
    "<linearGradient id='atlas-gradient' x1='0' y1='1' x2='0' y2='0'>",
    "<stop offset='0%' stop-color='#F2F2F2'/><stop offset='100%' stop-color='#111111'/>",
    "</linearGradient>",
    "<linearGradient id='author-gradient' x1='0' y1='1' x2='0' y2='0'>",
    "<stop offset='0%' stop-color='#F2F2F2'/><stop offset='100%' stop-color='#CC79A7'/>",
    "</linearGradient>",
    "<linearGradient id='pangi-gradient' x1='0' y1='1' x2='0' y2='0'>",
    "<stop offset='0%' stop-color='#F2F2F2'/><stop offset='100%' stop-color='#D9782D'/>",
    "</linearGradient>",
    "<linearGradient id='cap-gradient' x1='0' y1='1' x2='0' y2='0'>",
    "<stop offset='0%' stop-color='#F2F2F2'/><stop offset='100%' stop-color='#0096A6'/>",
    "</linearGradient>",
    "<linearGradient id='collection-gradient' x1='0' y1='1' x2='0' y2='0'>",
    "<stop offset='0%' stop-color='#D9D9D9'/><stop offset='50%' stop-color='#FFFFFF'/>",
    "<stop offset='100%' stop-color='#56B4E9'/></linearGradient>",
    "</defs>"
  )
}

postcap_legend_markup <- function(x, y = 18) {
  legend_specs <- tribble(
    ~title, ~gradient, ~low, ~mid, ~high,
    "Atlas metric", "atlas-gradient", "0", "", "1",
    "Author annotation concordance", "author-gradient", "0", "", "1",
    "PanGI exact match", "pangi-gradient", "0", "", "1",
    "CAP review", "cap-gradient", "0", "", "1",
    "Collection CLR (row z)", "collection-gradient", "-2", "0", "2"
  )
  pmap_chr(legend_specs, function(title, gradient, low, mid, high) {
    index <- match(title, legend_specs$title) - 1
    top <- y + index * 45
    middle <- top + 19
    paste0(
      "<g font-family='Helvetica' fill='#000000'>",
      "<text x='", x, "' y='", top, "' font-size='5.5' font-weight='bold'>",
      svg_escape(title), "</text>",
      "<rect x='", x, "' y='", top + 5, "' width='7' height='31' ",
      "fill='url(#", gradient, ")' stroke='#888888' stroke-width='0.25'/>",
      "<text x='", x + 9, "' y='", top + 9, "' font-size='4.5'>", high, "</text>",
      if (mid == "") "" else paste0(
        "<text x='", x + 9, "' y='", middle + 2,
        "' font-size='4.5'>", mid, "</text>"
      ),
      "<text x='", x + 9, "' y='", top + 36, "' font-size='4.5'>", low, "</text>",
      "</g>"
    )
  }) %>%
    paste(collapse = "\n")
}

axis_label_for <- function(feature) {
  dplyr::recode(
    feature,
    "WM" = "whole mucosa",
    "EPI_LP_MUSC" = "full thickness",
    "EPI_LP" = "epithelium and lamina propria",
    .default = feature
  )
}

# Prefer epithelial CSV for the shared column schema; all lineages share it.
build_column_key <- function(tiles) {
  group_levels <- c(
    "Atlas", "PanGI", "Author hierarchy", "CAP",
    "Collection method", "Dataset"
  )
  tiles %>%
    distinct(feature, group, column_index) %>%
    filter(!group %in% drop_groups) %>%
    arrange(column_index) %>%
    mutate(
      group = factor(as.character(group), levels = group_levels),
      column_index = row_number(),
      axis_label = axis_label_for(feature)
    )
}

postcap_sidecar_block <- function(
  result, y_offset, matrix_x, cell_width = postcap_cell_width
) {
  asset <- result$postcap_asset
  positions <- asset$leaf_positions %>%
    transmute(
      label_key,
      source_y = y,
      label_end_x = x + coalesce(text_length, 0)
    )
  tile_rows <- result$tiles %>%
    mutate(label_key = normalize_celltype_label(hgca_celltype_v1)) %>%
    left_join(positions, by = "label_key") %>%
    left_join(
      result$column_key %>% select(feature, group, column_index),
      by = c("feature", "group"),
      suffix = c("_old", "")
    )
  if (any(is.na(tile_rows$source_y))) {
    stop("Sidecar rows do not align with post-CAP SVG: ", result$spec$slug)
  }
  if (any(is.na(tile_rows$column_index))) {
    stop("Missing remapped column index for ", result$spec$slug)
  }
  row_pitch <- median(diff(sort(unique(positions$source_y))))
  tile_height <- row_pitch * 0.82
  tile_rows <- tile_rows %>%
    mutate(row_center_y = source_y - row_pitch / 2)
  connector_markup <- positions %>%
    mutate(
      row_center_y = source_y - row_pitch / 2,
      x1 = label_end_x + 3,
      x2 = matrix_x - 3
    ) %>%
    filter(x1 < x2) %>%
    transmute(
      markup = paste0(
        "<line x1='", sprintf("%.2f", x1),
        "' x2='", sprintf("%.2f", x2),
        "' y1='", sprintf("%.2f", y_offset + row_center_y),
        "' y2='", sprintf("%.2f", y_offset + row_center_y),
        "' stroke='#D2D2D2' stroke-width='0.45'/>"
      )
    ) %>%
    pull(markup) %>%
    paste(collapse = "\n")
  tile_markup <- pmap_chr(
    tile_rows %>%
      select(
        feature, group, value, column_index, row_center_y
      ),
    function(feature, group, value, column_index, row_center_y) {
      x <- matrix_x + (column_index - 1 + 0.04) * cell_width
      paste0(
        "<rect x='", sprintf("%.2f", x), "' y='",
        sprintf("%.2f", y_offset + row_center_y - tile_height / 2),
        "' width='", sprintf("%.2f", cell_width * 0.92),
        "' height='", sprintf("%.2f", tile_height),
        "' fill='", sidecar_tile_color(value, as.character(group)),
        "' stroke='#FFFFFF' stroke-width='0.25'/>"
      )
    }
  ) %>%
    paste(collapse = "\n")

  boundaries <- result$column_key %>%
    group_by(group) %>%
    summarise(first = min(column_index), .groups = "drop") %>%
    filter(first > 1)
  boundary_markup <- paste0(
    "<line x1='",
    sprintf("%.2f", matrix_x + (boundaries$first - 1) * cell_width),
    "' x2='",
    sprintf("%.2f", matrix_x + (boundaries$first - 1) * cell_width),
    "' y1='", sprintf("%.2f", y_offset + min(positions$source_y) - row_pitch),
    "' y2='", sprintf("%.2f", y_offset + max(positions$source_y)),
    "' stroke='#000000' stroke-width='0.6'/>",
    collapse = "\n"
  )
  title_markup <- paste0(
    "<text x='5' y='", y_offset + 12,
    "' font-family='Helvetica' font-size='8' font-weight='bold'>",
    svg_escape(result$spec$display), "</text>"
  )
  paste0(
    "<g transform='translate(0 ", y_offset, ")'>", asset$markup, "</g>\n",
    title_markup, "\n", connector_markup, "\n", tile_markup, "\n",
    boundary_markup
  )
}

postcap_axis_markup <- function(
  column_key, matrix_x, y, cell_width = postcap_cell_width
) {
  paste0(
    "<text x='",
    sprintf("%.2f", matrix_x + (column_key$column_index - 0.5) * cell_width),
    "' y='", sprintf("%.2f", y),
    "' transform='rotate(90 ",
    sprintf("%.2f", matrix_x + (column_key$column_index - 0.5) * cell_width),
    " ", sprintf("%.2f", y),
    ")' font-family='Helvetica' font-size='",
    postcap_column_label_size, "' text-anchor='start'>",
    svg_escape(column_key$axis_label), "</text>",
    collapse = "\n"
  )
}

postcap_tree_axis_markup <- function(x, y) {
  level_x <- c(33.39, 55.10, 76.82, 98.53, 120.25) * postcap_branch_x_scale
  paste0(
    "<g font-family='Helvetica' fill='#000000'>",
    "<line x1='", min(level_x), "' x2='", max(level_x),
    "' y1='", y, "' y2='", y,
    "' transform='translate(", x, " 0)' stroke='#555555' stroke-width='0.8'/>",
    paste0(
      "<line x1='", x + level_x, "' x2='", x + level_x,
      "' y1='", y - 3, "' y2='", y + 3,
      "' stroke='#555555' stroke-width='0.8'/>",
      "<text x='", x + level_x, "' y='", y + 17,
      "' font-size='11' text-anchor='middle'>", seq_along(level_x), "</text>",
      collapse = ""
    ),
    "<text x='", x + mean(range(level_x)), "' y='", y + 36,
    "' font-size='11' text-anchor='middle'>Taxonomy level</text>",
    "</g>"
  )
}

write_postcap_sidecar_svg <- function(results, path, stacked = FALSE) {
  results <- compact(results)
  if (length(results) == 0) stop("No sidecars available for SVG composition")
  column_key <- results[[1]]$column_key
  mismatched <- keep(results, function(result) {
    !identical(
      result$column_key %>% select(feature, group, column_index),
      column_key %>% select(feature, group, column_index)
    )
  })
  if (length(mismatched) > 0) {
    stop(
      "Sidecar column schemas differ for: ",
      paste(map_chr(mismatched, ~ .x$spec$slug), collapse = ", ")
    )
  }
  common_tree_width <- max(map_dbl(results, ~ .x$postcap_asset$tree_width))
  matrix_x <- common_tree_width + 7
  cell_width <- postcap_cell_width
  n_columns <- nrow(column_key)
  matrix_width <- n_columns * cell_width
  legend_x <- matrix_x + matrix_width + 13
  legend_width <- 105

  if (stacked) {
    results <- results[match(lineage_order, map_chr(results, ~ .x$spec$slug))]
    gap <- 8
    offsets <- c(
      0, head(cumsum(map_dbl(results, ~ .x$postcap_asset$height) + gap), -1)
    )
    content_height <- sum(map_dbl(results, ~ .x$postcap_asset$height)) +
      gap * (length(results) - 1)
  } else {
    offsets <- 0
    content_height <- results[[1]]$postcap_asset$height
  }
  axis_y <- content_height + 8
  axis_label_extent <- max(map_dbl(
    column_key$axis_label,
    ~ grid::convertWidth(
      grid::grobWidth(grid::textGrob(
        .x,
        gp = grid::gpar(
          fontfamily = "Helvetica", fontsize = postcap_column_label_size
        )
      )),
      "pt", valueOnly = TRUE
    )
  ))
  total_height <- axis_y + axis_label_extent + 10
  total_width <- legend_x + legend_width
  blocks <- map2_chr(
    results, offsets,
    ~ postcap_sidecar_block(.x, .y, matrix_x, cell_width)
  )
  axis <- postcap_axis_markup(column_key, matrix_x, axis_y, cell_width)
  tree_axis <- postcap_tree_axis_markup(0, axis_y)
  legend <- postcap_legend_markup(legend_x, 18)
  svg <- paste0(
    "<?xml version='1.0' encoding='UTF-8'?>\n",
    "<svg xmlns='http://www.w3.org/2000/svg' width='", total_width,
    "pt' height='", total_height, "pt' viewBox='0 0 ", total_width, " ",
    total_height, "'>\n",
    "<rect width='100%' height='100%' fill='#FFFFFF'/>\n",
    postcap_svg_defs(), "\n",
    paste(blocks, collapse = "\n"), "\n",
    tree_axis, "\n", axis, "\n", legend, "\n</svg>\n"
  )
  writeLines(svg, path, useBytes = TRUE)
  output_base <- tools::file_path_sans_ext(path)
  export_svg <- function(format, output_path, extra_args = character()) {
    status <- system2(
      "python",
      c(
        "-m", "cairosvg", shQuote(path), "-f", format,
        "-o", shQuote(output_path), extra_args
      ),
      env = "DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib"
    )
    if (!identical(status, 0L)) {
      warning("Could not export ", output_path, " with CairoSVG")
    }
  }
  export_svg("pdf", paste0(output_base, ".pdf"))
  export_svg("png", paste0(output_base, ".png"), c("-d", "300"))
  message("Saved ", output_base, ".{svg,pdf,png}")
}

build_result <- function(slug) {
  csv_path <- file.path(data_dir, paste0("hgca_arbol_sidecar_", slug, ".csv"))
  if (!file.exists(csv_path)) stop("Missing ", csv_path)
  tiles_raw <- readr::read_csv(csv_path, show_col_types = FALSE) %>%
    mutate(
      hgca_celltype_v1 = normalize_celltype_label(hgca_celltype_v1),
      group = as.character(group)
    )
  column_key <- build_column_key(tiles_raw)
  tiles <- tiles_raw %>%
    filter(!group %in% drop_groups) %>%
    inner_join(
      column_key %>% select(feature, group, column_index),
      by = c("feature", "group"),
      suffix = c("_old", "")
    ) %>%
    select(
      hgca_celltype_v1, feature, group, value, mean_clr, n_samples,
      leaf_x, column_index
    )
  list(
    spec = list(slug = slug, display = lineage_display[[slug]]),
    postcap_asset = read_postcap_svg_tree(slug),
    tiles = tiles,
    column_key = column_key
  )
}

results <- map(lineage_order, build_result)
names(results) <- lineage_order

# Shared column schema check / force shared key from first lineage after filter
shared_key <- results[[1]]$column_key
results <- map(results, function(result) {
  result$column_key <- shared_key
  result$tiles <- result$tiles %>%
    select(-column_index) %>%
    left_join(
      shared_key %>% select(feature, group, column_index),
      by = c("feature", "group")
    )
  if (any(is.na(result$tiles$column_index))) {
    stop("Could not align columns for ", result$spec$slug)
  }
  result
})

for (slug in lineage_order) {
  write_postcap_sidecar_svg(
    list(results[[slug]]),
    file.path(
      out_dir,
      paste0(
        "fig2_postcap_svg_sidecar_no_tissue_radial_clr_", slug, ".svg"
      )
    ),
    stacked = FALSE
  )
}

write_postcap_sidecar_svg(
  results,
  file.path(
    out_dir,
    "fig2_postcap_svg_sidecars_all_lineages_no_tissue_radial_clr.svg"
  ),
  stacked = TRUE
)

message(
  "Dropped groups: ", paste(drop_groups, collapse = ", "),
  "; retained columns: ", nrow(shared_key)
)
