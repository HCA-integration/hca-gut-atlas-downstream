#!/usr/bin/env Rscript
# Render PanGI-vs-HGCA ARBOL overlays from locally cached LODO label counts.
#
# Before final rendering, curate:
#   data/pangi_to_hgca_v1_crosswalk.csv
#
# Each included PanGI label needs either:
#   1. hgca_v1_label: a matching HGCA v1 leaf, or
#   2. parent_path: an exact ARBOL node path under which the PanGI-only label
#      should be injected.
#
# Duplicate a PanGI row when one broad PanGI label legitimately maps to more
# than one HGCA leaf. In that case, replace pangi_n_cells with apportioned
# counts; do not duplicate the full count across target leaves.

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidygraph)
  library(ggraph)
  library(svglite)
})

args <- commandArgs(trailingOnly = TRUE)
allow_partial <- "--allow-partial" %in% args
accept_current <- "--accept-current" %in% args

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Cannot determine script location")
script_path <- normalizePath(sub("^--file=", "", script_arg))
figure_dir <- normalizePath(file.path(dirname(script_path), ".."))
gca_root <- normalizePath(file.path(figure_dir, "..", ".."))

data_dir <- file.path(figure_dir, "data")
out_dir <- file.path(figure_dir, "out")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

crosswalk_path <- file.path(data_dir, "pangi_to_hgca_v1_crosswalk.csv")
hgca_counts_path <- file.path(data_dir, "hgca_v1_reference_label_counts.csv")
celltype_summary_path <- file.path(data_dir, "celltype_atlas_summary.csv")
composition_path <- file.path(
  data_dir, "celltype_compositional_enrichment_long.csv"
)
cap_summary_path <- file.path(data_dir, "cap_celltype_summary.csv")
dataset_counts_path <- file.path(data_dir, "dataset_celltype_counts_long.csv")
graph_path <- file.path(gca_root, "ARBOL", "taxonomy_rose_ggraph_v1.rds")
taxonomy_path <- file.path(gca_root, "ontology", "GCA_taxonomy_2026_CAP.csv")
arbol_source <- file.path(
  gca_root, "reference_mapping_benchmark", "src", "visualization", "arbol.R"
)

required_paths <- c(
  crosswalk_path, hgca_counts_path, celltype_summary_path,
  composition_path, cap_summary_path, dataset_counts_path, graph_path, taxonomy_path,
  arbol_source
)
missing_paths <- required_paths[!file.exists(required_paths)]
if (length(missing_paths) > 0) {
  stop("Missing required input(s):\n", paste(missing_paths, collapse = "\n"))
}

source(arbol_source)

crosswalk <- readr::read_csv(crosswalk_path, show_col_types = FALSE) %>%
  mutate(
    include = tolower(as.character(include)) %in% c("true", "t", "1", "yes", "y"),
    hgca_v1_label = na_if(str_squish(as.character(hgca_v1_label)), ""),
    parent_path = na_if(str_squish(as.character(parent_path)), ""),
    lineage = na_if(str_squish(as.character(lineage)), ""),
    mapped = !is.na(hgca_v1_label) | !is.na(parent_path),
    review_status = coalesce(as.character(review_status), "not_reviewed"),
    reviewed = str_detect(review_status, "^(approved|modified)")
  )

unresolved <- crosswalk %>% filter(include, !mapped | (!reviewed & !accept_current))
if (nrow(unresolved) > 0 && !allow_partial) {
  stop(
    nrow(unresolved),
    " included PanGI labels still need curator approval. ",
    "Edit ", crosswalk_path,
    " or rerun with --allow-partial for a clearly marked draft."
  )
}
if (nrow(unresolved) > 0) {
  warning("Partial draft: using ", nrow(unresolved), " unreviewed PanGI inferences")
}

crosswalk_use <- crosswalk %>% filter(include, mapped)
if (nrow(crosswalk_use) == 0) stop("No mapped PanGI labels are available")

hgca_counts <- readr::read_csv(hgca_counts_path, show_col_types = FALSE)
hgca_vec <- setNames(as.integer(hgca_counts$n_cells), hgca_counts$label)

mapped_counts <- crosswalk_use %>%
  filter(!is.na(hgca_v1_label)) %>%
  group_by(hgca_v1_label) %>%
  summarise(n_cells = sum(as.integer(pangi_n_cells)), .groups = "drop")
pangi_vec <- setNames(as.integer(mapped_counts$n_cells), mapped_counts$hgca_v1_label)

extra_rows <- crosswalk_use %>%
  filter(is.na(hgca_v1_label), !is.na(parent_path)) %>%
  transmute(
    parent_path = parent_path,
    label = pangi_level3_label,
    n_cells = as.integer(pangi_n_cells)
  )
if (nrow(extra_rows) > 0) {
  pangi_vec <- c(
    pangi_vec,
    setNames(extra_rows$n_cells, extra_rows$label)
  )
}

tax_graph <- readRDS(graph_path)
tax_graph <- ensure_required_node_fields(tax_graph)

taxonomy <- readr::read_csv(taxonomy_path, show_col_types = FALSE)
level_columns <- grep("^hgca_celltype_level[1-5]$", names(taxonomy), value = TRUE)
safe_path_token <- function(x) gsub("[^A-Za-z0-9]", "", as.character(x))
taxonomy_paths <- taxonomy %>%
  filter(!is.na(hgca_celltype_v1), hgca_celltype_v1 != "") %>%
  rowwise() %>%
  mutate(
    arbol_path = paste(
      safe_path_token(
        c_across(all_of(level_columns))[
          !is.na(c_across(all_of(level_columns))) &
            c_across(all_of(level_columns)) != ""
        ]
      ),
      collapse = "."
    )
  ) %>%
  ungroup() %>%
  distinct(hgca_celltype_v1, .keep_all = TRUE)
internal_label_paths <- setNames(
  taxonomy_paths$arbol_path,
  taxonomy_paths$hgca_celltype_v1
)

lineage_specs <- tribble(
  ~slug,         ~display,       ~subset_clade,
  "epithelial",  "Epithelial",   "root.Epithelial",
  "myeloid",     "Myeloid",      "root.Immune.Myeloid",
  "lymphoid",    "Lymphoid",     "root.Immune.Lymphoid",
  "stromal",     "Stromal",      "root.Stromal"
)

# Top-to-bottom leaf order in Fig2_Halfsize.ai, recovered from the matching
# post-CAP SVG assets under ARBOL/. Keeping these explicit prevents ggraph
# layout updates from silently changing manuscript row order.
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
    paste0(
      c("epithelial", "myeloid", "lymphoid", "stromal"),
      "_annotated_taxonomy_dendrogram_PostCAP_V1.svg"
    )
  ),
  c("epithelial", "myeloid", "lymphoid", "stromal")
)
missing_postcap <- postcap_svg_paths[!file.exists(postcap_svg_paths)]
if (length(missing_postcap) > 0) {
  stop("Missing post-CAP SVG asset(s):\n", paste(missing_postcap, collapse = "\n"))
}

celltype_summary <- readr::read_csv(
  celltype_summary_path, show_col_types = FALSE
)
composition_enrichment <- readr::read_csv(
  composition_path, show_col_types = FALSE
)
cap_summary <- readr::read_csv(cap_summary_path, show_col_types = FALSE)
dataset_counts <- readr::read_csv(
  dataset_counts_path, show_col_types = FALSE
)

# Canonical sidecar column schema across all lineages. Per-lineage NA filtering
# previously dropped Author match level 4 outside epithelial and shifted every
# later column in that panel relative to the stacked figure.
shared_metric_feature_order <- c(
  "Cells", "Datasets", "Samples", "Donors", "Rare",
  "LODO F1",
  "PanGI exact match",
  "Author match level 1", "Author match level 2", "Author match level 3",
  "Author match level 4", "Author match level 5",
  "Atlas increased resolution vs author", "Same resolution as author",
  "Atlas reduced resolution vs author", "Changed branch from author",
  "CAP votes", "CAP agreement", "CAP split/merge", "CAP uncertain"
)
shared_metric_keep <- shared_metric_feature_order[
  vapply(
    shared_metric_feature_order,
    function(feature) {
      source_col <- case_when(
        feature == "Cells" ~ "n_cells",
        feature == "Datasets" ~ "n_datasets",
        feature == "Samples" ~ "n_samples",
        feature == "Donors" ~ "n_donors",
        feature == "Rare" ~ "rare_lt_0_1pct",
        feature == "LODO F1" ~ "lodo_f1",
        feature == "PanGI exact match" ~ NA_character_,
        feature == "Author match level 1" ~ "author_level1_match_fraction",
        feature == "Author match level 2" ~ "author_level2_match_fraction",
        feature == "Author match level 3" ~ "author_level3_match_fraction",
        feature == "Author match level 4" ~ "author_level4_match_fraction",
        feature == "Author match level 5" ~ "author_level5_match_fraction",
        feature == "Atlas increased resolution vs author" ~
          "atlas_increased_resolution_fraction",
        feature == "Same resolution as author" ~
          "atlas_same_resolution_fraction",
        feature == "Atlas reduced resolution vs author" ~
          "atlas_reduced_resolution_fraction",
        feature == "Changed branch from author" ~
          "atlas_changed_branch_fraction",
        feature == "CAP votes" ~ "cap_vote_count",
        feature == "CAP agreement" ~ "cap_agreement_fraction",
        feature == "CAP split/merge" ~ "cap_split_merge_fraction",
        feature == "CAP uncertain" ~ "cap_uncertain_fraction",
        TRUE ~ NA_character_
      )
      if (feature == "PanGI exact match") return(TRUE)
      if (is.na(source_col)) return(FALSE)
      if (str_starts(feature, "CAP ")) {
        return(any(!is.na(cap_summary[[source_col]])))
      }
      any(!is.na(celltype_summary[[source_col]]))
    },
    logical(1)
  )
]
shared_composition_key <- composition_enrichment %>%
  distinct(feature = annotation_level, group = annotation_group, level_order) %>%
  mutate(
    group = factor(
      group,
      levels = c("Tissue", "Collection method", "Radial layer")
    )
  ) %>%
  arrange(group, level_order)
shared_dataset_order <- dataset_counts %>%
  group_by(dataset_id) %>%
  summarise(total = sum(n_cells), .groups = "drop") %>%
  arrange(desc(total)) %>%
  pull(dataset_id)
shared_sidecar_column_key <- bind_rows(
  tibble(
    feature = shared_metric_keep,
    group = case_when(
      feature == "PanGI exact match" ~ "PanGI",
      str_detect(feature, "^Author match level|author$") ~ "Author hierarchy",
      str_starts(feature, "CAP ") ~ "CAP",
      TRUE ~ "Atlas"
    ),
    level_order = NA_real_
  ),
  shared_composition_key,
  tibble(
    feature = shared_dataset_order,
    group = "Dataset",
    level_order = NA_real_
  )
) %>%
  mutate(
    column_index = row_number(),
    column_y = -6.0 - 0.72 * (column_index - 1),
    axis_label = recode(
      feature,
      "WM" = "whole mucosa",
      "EPI_LP_MUSC" = "full thickness",
      "EPI_LP" = "epithelium and lamina propria",
      .default = feature
    )
  )
message(
  "Shared sidecar columns: ", nrow(shared_sidecar_column_key),
  " (author levels kept: ",
  paste(
    shared_metric_keep[str_detect(shared_metric_keep, "^Author match level")],
    collapse = ", "
  ),
  ")"
)

rescale01 <- function(x, log_transform = FALSE) {
  x <- as.numeric(x)
  if (log_transform) x <- log10(x + 1)
  max_x <- max(x, na.rm = TRUE)
  if (!is.finite(max_x) || max_x == 0) return(rep(0, length(x)))
  x / max_x
}

# Taxonomy / ARBOL occasionally embeds newlines in display labels
# (currently MO DC). Atlas metadata uses a single-line form, so normalize
# before any join on hgca_celltype_v1.
normalize_celltype_label <- function(x) {
  str_squish(str_replace_all(as.character(x), "[\\r\\n]+", " "))
}

create_unique_row_tree_layout <- function(graph, reference_labels = NULL) {
  # ggraph's tree layout can assign the same x coordinate to distinct terminal
  # nodes. Use dendrogram leaf order to give every leaf one row, retain the
  # tree layout's true-depth y coordinate, then center each parent over its
  # direct children. This uses ggraph/tidygraph data only.
  tree_layout <- create_layout(graph, layout = "tree")
  if (is.null(reference_labels)) {
    dendrogram_layout <- create_layout(graph, layout = "dendrogram")
    leaf_order <- dendrogram_layout %>%
      as_tibble() %>%
      filter(numChildren == 0) %>%
      arrange(x, name) %>%
      pull(.ggraph.orig_index)
  } else {
    leaf_nodes <- tree_layout %>%
      as_tibble() %>%
      filter(numChildren == 0) %>%
      transmute(
        original_index = .ggraph.orig_index,
        label_key = normalize_celltype_label(hgca_celltype_v1_majority)
      )
    reference_keys <- normalize_celltype_label(reference_labels)
    missing <- setdiff(reference_keys, leaf_nodes$label_key)
    extra <- setdiff(leaf_nodes$label_key, reference_keys)
    if (length(missing) > 0 || length(extra) > 0) {
      stop(
        "Reference ARBOL leaf order does not match graph leaves. Missing: ",
        paste(missing, collapse = ", "),
        "; extra: ", paste(extra, collapse = ", ")
      )
    }
    leaf_order <- leaf_nodes$original_index[
      match(reference_keys, leaf_nodes$label_key)
    ]
  }

  original_indices <- tree_layout$.ggraph.orig_index
  x_by_index <- setNames(
    rep(NA_real_, length(original_indices)),
    as.character(original_indices)
  )
  # coord_flip places larger layout x values at the top of the rendered tree.
  x_by_index[as.character(leaf_order)] <- rev(seq_along(leaf_order))
  edges <- graph %E>% as_tibble() %>% select(from, to)
  pending <- setdiff(original_indices, leaf_order)

  while (length(pending) > 0) {
    ready <- pending[vapply(pending, function(parent) {
      children <- edges$to[edges$from == parent]
      length(children) > 0 &&
        all(!is.na(x_by_index[as.character(children)]))
    }, logical(1))]
    if (length(ready) == 0) {
      stop("Could not derive unique sidecar rows from the tidygraph hierarchy")
    }
    for (parent in ready) {
      children <- edges$to[edges$from == parent]
      x_by_index[as.character(parent)] <- mean(
        x_by_index[as.character(children)]
      )
    }
    pending <- setdiff(pending, ready)
  }

  tree_layout$x <- unname(x_by_index[as.character(original_indices)])
  leaf_rows <- tree_layout$x[tree_layout$numChildren == 0]
  if (anyDuplicated(leaf_rows)) {
    stop("Sidecar row layout still contains duplicate leaf coordinates")
  }
  tree_layout
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

  # Pull only the actual tree edges and leaf labels from the source SVG. This
  # excludes the source title and repeated lineage legend while preserving the
  # post-CAP branch geometry, colors, typography, and native leaf coordinates.
  polyline_nodes <- xml2::xml_find_all(doc, ".//*[local-name()='polyline']")
  line_nodes <- xml2::xml_find_all(doc, ".//*[local-name()='line']")
  line_nodes <- line_nodes[
    svg_number(xml2::xml_attr(line_nodes, "x1")) <= 130 &
      svg_number(xml2::xml_attr(line_nodes, "x2")) <= 130
  ]
  # The source svglite CSS supplies fill:none. Reinstate it after extracting
  # the elements so polylines cannot render as closed black polygons.
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
  if (group == "Tissue") {
    return(interpolate_svg_color(
      value, c("#D9D9D9", "#FFFFFF", "#009E73"), c(-2.25, 2.25)
    ))
  }
  if (group == "Collection method") {
    return(interpolate_svg_color(
      value, c("#D9D9D9", "#FFFFFF", "#56B4E9"), c(-2.25, 2.25)
    ))
  }
  if (group == "Radial layer") {
    return(interpolate_svg_color(
      value, c("#D9D9D9", "#FFFFFF", "#7B3294"), c(-2.25, 2.25)
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
    "<linearGradient id='tissue-gradient' x1='0' y1='1' x2='0' y2='0'>",
    "<stop offset='0%' stop-color='#D9D9D9'/><stop offset='50%' stop-color='#FFFFFF'/>",
    "<stop offset='100%' stop-color='#009E73'/></linearGradient>",
    "<linearGradient id='collection-gradient' x1='0' y1='1' x2='0' y2='0'>",
    "<stop offset='0%' stop-color='#D9D9D9'/><stop offset='50%' stop-color='#FFFFFF'/>",
    "<stop offset='100%' stop-color='#56B4E9'/></linearGradient>",
    "<linearGradient id='radial-gradient' x1='0' y1='1' x2='0' y2='0'>",
    "<stop offset='0%' stop-color='#D9D9D9'/><stop offset='50%' stop-color='#FFFFFF'/>",
    "<stop offset='100%' stop-color='#7B3294'/></linearGradient>",
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
    "Tissue CLR (row z)", "tissue-gradient", "-2", "0", "2",
    "Collection CLR (row z)", "collection-gradient", "-2", "0", "2",
    "Radial CLR (row z)", "radial-gradient", "-2", "0", "2"
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

postcap_sidecar_block <- function(
  result, y_offset, matrix_x, cell_width = postcap_cell_width,
  show_title = TRUE
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
    left_join(positions, by = "label_key")
  if (any(is.na(tile_rows$source_y))) {
    stop("Sidecar rows do not align with post-CAP SVG: ", result$spec$slug)
  }
  row_pitch <- median(diff(sort(unique(positions$source_y))))
  tile_height <- row_pitch * 0.82
  # SVG text y is the baseline, not its visual center. Shifting by half the
  # native row pitch aligns tile centers to the optical center of each label.
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
    tile_rows,
    function(
      hgca_celltype_v1, feature, group, value, mean_clr, n_samples,
      leaf_x, column_index, label_key, source_y, label_end_x, row_center_y
    ) {
      # Center the 92%-width tile within its full column so the rotated label
      # baseline and visible tile center share exactly the same x coordinate.
      x <- matrix_x + (column_index - 1 + 0.04) * cell_width
      paste0(
        "<rect x='", sprintf("%.2f", x), "' y='",
        sprintf("%.2f", y_offset + row_center_y - tile_height / 2),
        "' width='", sprintf("%.2f", cell_width * 0.92),
        "' height='", sprintf("%.2f", tile_height),
        "' fill='", sidecar_tile_color(value, group),
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
  title_markup <- if (show_title) {
    paste0(
      "<text x='5' y='", y_offset + 12,
      "' font-family='Helvetica' font-size='8' font-weight='bold'>",
      svg_escape(result$spec$display), "</text>"
    )
  } else {
    ""
  }
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
  level_x <- c(33.39, 55.10, 76.82, 98.53, 120.25) *
    postcap_branch_x_scale
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
  column_key <- shared_sidecar_column_key
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
    order <- c("lymphoid", "stromal", "epithelial", "myeloid")
    results <- results[match(order, map_chr(results, ~ .x$spec$slug))]
    gap <- 8
    offsets <- c(0, head(cumsum(map_dbl(results, ~ .x$postcap_asset$height) + gap), -1))
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

save_aligned_sidecar <- function(spec) {
  lineage_graph <- subset_to_clade(tax_graph, spec$subset_clade)
  lineage_graph <- ensure_required_node_fields(lineage_graph)
  # Match the original HGCA v1 hierarchy figures: true taxonomy depth on the
  # tree panel, while leaf x-coordinates define the aligned heatmap rows.
  layout <- create_unique_row_tree_layout(
    lineage_graph, reference_leaf_orders[[spec$slug]]
  )
  leaves <- layout %>%
    as_tibble() %>%
    filter(numChildren == 0) %>%
    transmute(
      hgca_celltype_v1 = normalize_celltype_label(hgca_celltype_v1_majority),
      leaf_x = x,
      leaf_y = y,
      label_y = -5.55
    )
  if (anyDuplicated(leaves$hgca_celltype_v1)) {
    stop(
      "Normalized leaf labels are not unique for ", spec$slug, ": ",
      paste(
        leaves$hgca_celltype_v1[duplicated(leaves$hgca_celltype_v1)],
        collapse = ", "
      )
    )
  }

  pangi_presence <- mapped_counts %>%
    transmute(
      hgca_celltype_v1 = hgca_v1_label,
      pangi_present = as.integer(n_cells > 0)
    )
  metrics <- leaves %>%
    left_join(celltype_summary, by = "hgca_celltype_v1") %>%
    left_join(cap_summary, by = "hgca_celltype_v1") %>%
    left_join(pangi_presence, by = "hgca_celltype_v1") %>%
    mutate(
      pangi_present = coalesce(pangi_present, 0L),
      across(c(rare_lt_0_1pct, one_dataset_only), ~ as.integer(coalesce(.x, FALSE)))
    ) %>%
    transmute(
      hgca_celltype_v1,
      leaf_x,
      Cells = rescale01(n_cells, log_transform = TRUE),
      Datasets = rescale01(n_datasets),
      Samples = rescale01(n_samples, log_transform = TRUE),
      Donors = rescale01(n_donors, log_transform = TRUE),
      Rare = rare_lt_0_1pct,
      # Already 0–1 support-weighted mean per-class F1 from newest v1 LODO.
      `LODO F1` = as.numeric(lodo_f1),
      `PanGI exact match` = pangi_present,
      `Author match level 1` = author_level1_match_fraction,
      `Author match level 2` = author_level2_match_fraction,
      `Author match level 3` = author_level3_match_fraction,
      `Author match level 4` = author_level4_match_fraction,
      `Author match level 5` = author_level5_match_fraction,
      `Atlas increased resolution vs author` = atlas_increased_resolution_fraction,
      `Same resolution as author` = atlas_same_resolution_fraction,
      `Atlas reduced resolution vs author` = atlas_reduced_resolution_fraction,
      `Changed branch from author` = atlas_changed_branch_fraction,
      `CAP votes` = rescale01(cap_vote_count),
      `CAP agreement` = cap_agreement_fraction,
      `CAP split/merge` = cap_split_merge_fraction,
      `CAP uncertain` = cap_uncertain_fraction
    ) %>%
    pivot_longer(
      -c(hgca_celltype_v1, leaf_x),
      names_to = "feature",
      values_to = "value"
    ) %>%
    mutate(
      group = case_when(
        feature == "PanGI exact match" ~ "PanGI",
        str_detect(
          feature,
          "^Author match level|author$"
        ) ~ "Author hierarchy",
        str_starts(feature, "CAP ") ~ "CAP",
        TRUE ~ "Atlas"
      )
    ) %>%
    filter(feature %in% shared_metric_keep)

  composition_tiles <- composition_enrichment %>%
    inner_join(
      leaves %>% select(hgca_celltype_v1, leaf_x),
      by = "hgca_celltype_v1"
    ) %>%
    transmute(
      hgca_celltype_v1,
      leaf_x,
      feature = annotation_level,
      value = row_z,
      mean_clr,
      n_samples,
      level_order,
      group = annotation_group
    ) %>%
    mutate(
      group = factor(
        group,
        levels = c("Tissue", "Collection method", "Radial layer")
      )
    ) %>%
    arrange(group, level_order)

  dataset_tiles <- leaves %>%
    crossing(feature = shared_dataset_order) %>%
    left_join(
      dataset_counts %>%
        transmute(
          hgca_celltype_v1,
          feature = dataset_id,
          value = as.numeric(n_cells > 0)
        ),
      by = c("hgca_celltype_v1", "feature")
    ) %>%
    mutate(value = coalesce(value, 0), group = "Dataset")

  column_key <- shared_sidecar_column_key
  tiles <- leaves %>%
    select(hgca_celltype_v1, leaf_x) %>%
    crossing(column_key %>% select(feature, group, column_index, column_y, axis_label)) %>%
    left_join(
      bind_rows(metrics, composition_tiles, dataset_tiles) %>%
        select(
          hgca_celltype_v1, feature, group, value, mean_clr, n_samples
        ),
      by = c("hgca_celltype_v1", "feature", "group")
    )
  missing_columns <- setdiff(column_key$feature, unique(tiles$feature))
  if (length(missing_columns) > 0) {
    stop(
      "Sidecar for ", spec$slug,
      " is missing shared columns: ", paste(missing_columns, collapse = ", ")
    )
  }
  group_boundaries <- column_key %>%
    group_by(group) %>%
    summarise(boundary = min(column_y) - 0.36, .groups = "drop") %>%
    filter(group != "Dataset")

  plot <- ggraph(layout) +
    geom_edge_diagonal2(
      color = "#A6A6A6", linewidth = 0.5, alpha = 0.85,
      strength = 0.85, n = 60, lineend = "round"
    ) +
    geom_segment(
      data = leaves,
      aes(x = leaf_x, xend = leaf_x, y = leaf_y, yend = label_y),
      inherit.aes = FALSE,
      color = "#C7C7C7", linewidth = 0.3
    ) +
    geom_text(
      data = leaves,
      aes(
        x = leaf_x, y = label_y,
        label = hgca_celltype_v1
      ),
      inherit.aes = FALSE,
      hjust = 1, vjust = 0.5,
      family = "Helvetica", fontface = "bold", size = 2.6, color = "black"
    ) +
    geom_node_text(
      aes(
        filter = numChildren > 0,
        label = un_camel_case(
          ifelse(
            grepl("\\|", name),
            sub(".*\\|", "", name),
            sub(".*\\.", "", name)
          )
        )
      ),
      nudge_y = 0.12, hjust = 1, vjust = 0.5,
      family = "Helvetica", fontface = "italic", size = 1.8, color = "#666666",
      check_overlap = TRUE
    ) +
    geom_tile(
      data = tiles %>% filter(group == "Atlas"),
      aes(x = leaf_x, y = column_y, fill = value),
      inherit.aes = FALSE,
      width = 0.84, height = 0.68,
      color = "white", linewidth = 0.08
    ) +
    scale_fill_gradient(
      low = "#F2F2F2", high = "#111111", limits = c(0, 1),
      name = "Atlas metric",
      guide = guide_colorbar(
        barheight = grid::unit(8, "mm"), barwidth = grid::unit(2, "mm")
      )
    ) +
    ggnewscale::new_scale_fill() +
    geom_tile(
      data = tiles %>% filter(group == "PanGI"),
      aes(x = leaf_x, y = column_y, fill = value),
      inherit.aes = FALSE,
      width = 0.84, height = 0.68,
      color = "white", linewidth = 0.08
    ) +
    scale_fill_gradient(
      low = "#F2F2F2", high = "#D9782D", limits = c(0, 1),
      na.value = "white", name = "PanGI exact match",
      guide = guide_colorbar(
        barheight = grid::unit(8, "mm"), barwidth = grid::unit(2, "mm")
      )
    ) +
    ggnewscale::new_scale_fill() +
    geom_tile(
      data = tiles %>% filter(group == "Author hierarchy"),
      aes(x = leaf_x, y = column_y, fill = value),
      inherit.aes = FALSE,
      width = 0.84, height = 0.68,
      color = "white", linewidth = 0.08
    ) +
    scale_fill_gradient(
      low = "#F2F2F2", high = "#CC79A7", limits = c(0, 1),
      na.value = "white", name = "Author annotation concordance",
      guide = guide_colorbar(
        barheight = grid::unit(8, "mm"), barwidth = grid::unit(2, "mm")
      )
    ) +
    ggnewscale::new_scale_fill() +
    geom_tile(
      data = tiles %>% filter(group == "CAP"),
      aes(x = leaf_x, y = column_y, fill = value),
      inherit.aes = FALSE,
      width = 0.84, height = 0.68,
      color = "white", linewidth = 0.08
    ) +
    scale_fill_gradient(
      low = "#F2F2F2", high = "#0096A6", limits = c(0, 1),
      na.value = "white", name = "CAP review",
      guide = guide_colorbar(
        barheight = grid::unit(8, "mm"), barwidth = grid::unit(2, "mm")
      )
    ) +
    ggnewscale::new_scale_fill() +
    geom_tile(
      data = tiles %>% filter(group == "Tissue"),
      aes(x = leaf_x, y = column_y, fill = value),
      inherit.aes = FALSE,
      width = 0.84, height = 0.68,
      color = "white", linewidth = 0.08
    ) +
    scale_fill_gradient2(
      low = "#D9D9D9", mid = "white", high = "#009E73",
      midpoint = 0, limits = c(-2.25, 2.25), oob = scales::squish,
      name = "Tissue CLR\n(row z-score)",
      guide = guide_colorbar(
        barheight = grid::unit(11, "mm"), barwidth = grid::unit(2, "mm")
      )
    ) +
    ggnewscale::new_scale_fill() +
    geom_tile(
      data = tiles %>% filter(group == "Collection method"),
      aes(x = leaf_x, y = column_y, fill = value),
      inherit.aes = FALSE,
      width = 0.84, height = 0.68,
      color = "white", linewidth = 0.08
    ) +
    scale_fill_gradient2(
      low = "#D9D9D9", mid = "white", high = "#56B4E9",
      midpoint = 0, limits = c(-2.25, 2.25), oob = scales::squish,
      name = "Collection CLR\n(row z-score)",
      guide = guide_colorbar(
        barheight = grid::unit(11, "mm"), barwidth = grid::unit(2, "mm")
      )
    ) +
    ggnewscale::new_scale_fill() +
    geom_tile(
      data = tiles %>% filter(group == "Radial layer"),
      aes(x = leaf_x, y = column_y, fill = value),
      inherit.aes = FALSE,
      width = 0.84, height = 0.68,
      color = "white", linewidth = 0.08
    ) +
    scale_fill_gradient2(
      low = "#D9D9D9", mid = "white", high = "#7B3294",
      midpoint = 0, limits = c(-2.25, 2.25), oob = scales::squish,
      name = "Radial CLR\n(row z-score)",
      guide = guide_colorbar(
        barheight = grid::unit(11, "mm"), barwidth = grid::unit(2, "mm")
      )
    ) +
    ggnewscale::new_scale_fill() +
    geom_tile(
      data = tiles %>% filter(group == "Dataset"),
      aes(x = leaf_x, y = column_y, fill = value),
      inherit.aes = FALSE,
      width = 0.84, height = 0.68,
      color = "white", linewidth = 0.08
    ) +
    scale_fill_gradient(
      low = "#F2F2F2", high = "#3A6EA5", limits = c(0, 1),
      name = "Dataset presence", guide = "none"
    ) +
    geom_hline(
      data = group_boundaries,
      aes(yintercept = boundary),
      inherit.aes = FALSE,
      color = "black", linewidth = 0.45
    ) +
    scale_y_reverse(
      breaks = column_key$column_y,
      labels = column_key$axis_label,
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    coord_flip(clip = "off") +
    labs(
      title = paste0(
        "HGCA v1 hierarchy and atlas evidence sidecar: ", spec$display
      )
    ) +
    theme_void(base_family = "Helvetica", base_size = 7) +
    theme(
      text = element_text(color = "black"),
      axis.text.x = element_text(
        angle = 90, hjust = 1, vjust = 0.5, size = 7, color = "black"
      ),
      axis.ticks.x = element_line(color = "black", linewidth = 0.25),
      plot.title = element_text(
        hjust = 0, face = "bold", size = 8, margin = margin(b = 5)
      ),
      legend.position = "right",
      legend.title = element_text(size = 5.5),
      legend.text = element_text(size = 4.5),
      legend.spacing.y = grid::unit(1.5, "mm"),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(t = 5, r = 8, b = 5, l = 55)
    )

  n_leaves <- nrow(leaves)
  width <- 15
  height <- max(4.2, 1.8 + 0.25 * n_leaves)
  base <- file.path(
    out_dir, paste0("fig2_hgca_arbol_sidecar_", spec$slug)
  )
  ggsave(paste0(base, ".pdf"), plot, width = width, height = height,
         device = cairo_pdf, bg = "white")
  ggsave(paste0(base, ".svg"), plot, width = width, height = height,
         device = svglite::svglite, bg = "white")
  ggsave(paste0(base, ".png"), plot, width = width, height = height, dpi = 300,
         bg = "white")
  readr::write_csv(
    tiles %>%
      select(
        hgca_celltype_v1, feature, group, value, mean_clr, n_samples,
        leaf_x, column_index
      ),
    file.path(data_dir, paste0("hgca_arbol_sidecar_", spec$slug, ".csv"))
  )
  message("Saved ", base, ".{pdf,svg,png}")

  postcap_result <- list(
    spec = spec,
    postcap_asset = read_postcap_svg_tree(spec$slug),
    tiles = tiles %>%
      select(
        hgca_celltype_v1, feature, group, value, mean_clr, n_samples,
        leaf_x, column_index
      ),
    column_key = column_key
  )
  postcap_base <- file.path(
    out_dir, paste0("fig2_postcap_svg_sidecar_", spec$slug)
  )
  write_postcap_sidecar_svg(
    list(postcap_result), paste0(postcap_base, ".svg")
  )
  postcap_result
}

save_overlay <- function(spec) {
  extra_lineage <- extra_rows
  if ("lineage" %in% names(crosswalk_use)) {
    extra_labels <- crosswalk_use %>%
      filter(is.na(hgca_v1_label), lineage == spec$display) %>%
      pull(pangi_level3_label)
    extra_lineage <- extra_rows %>% filter(label %in% extra_labels)
  }

  overlay <- create_arbol_overlay(
    graph = tax_graph,
    original_labels_vec = pangi_vec,
    transferred_labels_vec = hgca_vec,
    lineage_name = spec$display,
    subset_clade = spec$subset_clade,
    title = paste0("Cell types added by HGCA v1 relative to PanGI: ", spec$display),
    original_label = "PanGI",
    reference_label = "HGCA",
    original_color = "#D9782D",
    reference_color = "#3A6EA5",
    absent_color = "#E0E0E0",
    extra_leaves = extra_lineage %>% select(parent_path, label),
    extra_only_labels = extra_lineage$label,
    extra_only_label = "PanGI only",
    extra_only_color = "#999999",
    drop_absent_in_both = FALSE,
    internal_label_paths = internal_label_paths
  )

  suffix <- if (allow_partial && nrow(unresolved) > 0) "_partial_draft" else ""
  base <- file.path(
    out_dir,
    paste0("fig2_pangi_hgca_arbol_", spec$slug, suffix)
  )
  ggsave(paste0(base, ".pdf"), overlay$plot,
         width = overlay$width, height = overlay$height, device = cairo_pdf)
  ggsave(paste0(base, ".svg"), overlay$plot,
         width = overlay$width, height = overlay$height, device = svglite::svglite)
  ggsave(paste0(base, ".png"), overlay$plot,
         width = overlay$width, height = overlay$height, dpi = 300)
  readr::write_csv(
    overlay$leaves_df,
    file.path(data_dir, paste0("pangi_hgca_arbol_", spec$slug, "_leaves.csv"))
  )
  message("Saved ", base, ".{pdf,svg,png}")
}

postcap_results <- purrr::pmap(lineage_specs, function(slug, display, subset_clade) {
  tryCatch(
    {
      spec <- list(slug = slug, display = display, subset_clade = subset_clade)
      save_overlay(spec)
      save_aligned_sidecar(spec)
    },
    error = function(e) warning(display, " ARBOL skipped: ", conditionMessage(e))
  )
})

postcap_results <- compact(postcap_results)
if (length(postcap_results) == nrow(lineage_specs)) {
  write_postcap_sidecar_svg(
    postcap_results,
    file.path(out_dir, "fig2_postcap_svg_sidecars_all_lineages.svg"),
    stacked = TRUE
  )
}
