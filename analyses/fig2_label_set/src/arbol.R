# ARBOL Visualization Functions
#
# Extracted from baseline_pilot_myeloid/notebooks/06_arbol_visualization.ipynb
# Refactored for multi-hgca_celltype_level1 reusability
#
# Key functions:
# - subset_to_clade(): Subset taxonomy to specific hgca_celltype_level1
# - calculate_perclass_accuracy(): Load predictions and calculate accuracy
# - create_arbol_plot(): Generate ARBOL dendrogram
# - calculate_plot_size(): Dynamic sizing based on node count

library(tidyverse)
library(tidygraph)
library(ggraph)
library(jsonlite)
library(purrr)
library(stringr)
# Ensure required node attributes exist on a taxonomy graph
ensure_required_node_fields <- function(graph) {
  # Compute numChildren if missing using edge out-degree
  nodes_df <- graph %N>% as_tibble()
  need_num_children <- !("numChildren" %in% colnames(nodes_df))
  need_hgca_celltype_level1_majority <- !("hgca_celltype_level1_majority" %in% colnames(nodes_df))
  need_leaf_label <- !("hgca_celltype_v1_majority" %in% colnames(nodes_df))

  if (need_num_children) {
    # Map edge indices (from) to node names, then count by name
    edges_tbl <- as_tibble(graph, active = "edges")
    nodes_tbl <- graph %N>% as_tibble() %>% mutate(.row_id = dplyr::row_number())
    edge_counts <- edges_tbl %>%
      left_join(nodes_tbl %>% select(.row_id, name) %>% rename(from = .row_id, from_name = name), by = "from") %>%
      count(from_name, name = "numChildren")
    graph <- graph %N>% left_join(edge_counts, by = c("name" = "from_name")) %>%
      mutate(numChildren = if_else(is.na(numChildren), 0L, as.integer(numChildren)))
  }

  if (need_hgca_celltype_level1_majority) {
    graph <- graph %N>% mutate(
      hgca_celltype_level1_majority = dplyr::case_when(
        !is.na(hgca_celltype_level1) ~ hgca_celltype_level1,
        TRUE ~ {
          # derive from dot-path name (root.hgca_celltype_level1....)
          parts <- str_split(name, fixed("."))
          vapply(parts, function(x) if (length(x) >= 2) x[2] else "root", character(1))
        }
      )
    )
  }

  if (need_leaf_label) {
    nodes_df2 <- graph %N>% as_tibble()
    # Accept several alternate columns from different builders
    alt_cols <- c("hgca_celltype_v1_majority", "closest_label", "hgca_celltype_v1", "leaf", "label")
    present <- intersect(alt_cols, colnames(nodes_df2))
    co <- if (length(present) > 0) do.call(coalesce, nodes_df2[present]) else rep(NA_character_, nrow(nodes_df2))
    # Derive label from name using either '|' or '.' hierarchy separators
    derived <- ifelse(grepl("\\|", nodes_df2$name), sub('.*\\|', '', nodes_df2$name), sub('.*\\.', '', nodes_df2$name))
    fill_leaf <- ifelse(nodes_df2$numChildren == 0, ifelse(!is.na(co), as.character(co), derived), NA_character_)
    tmp <- tibble(name = nodes_df2$name, leaf_tmp = fill_leaf)
    # Create the column first (as NA) to allow coalesce when it doesn't exist
    graph <- graph %N>% mutate(hgca_celltype_v1_majority = NA_character_) %>%
      left_join(tmp, by = c("name" = "name")) %>%
      mutate(hgca_celltype_v1_majority = dplyr::coalesce(hgca_celltype_v1_majority, leaf_tmp)) %>%
      select(-leaf_tmp)
  }

  graph
}


# Function to subset graph to a specific clade
subset_to_clade <- function(graph, clade_name) {
  # Find all nodes descended from the specified clade
  # Uses name hierarchy matching (e.g., "root.Immune.Myeloid")
  
  node_data <- graph %N>% as_tibble()
  # Normalize separators to '.' and create a clean label from last token
  node_data <- node_data %>% mutate(
    name_norm = gsub("\\|", ".", name),
    label_clean = ifelse(grepl("\\|", name), sub('.*\\|', '', name), sub('.*\\.', '', name))
  )
  clade_name_norm <- gsub("\\|", ".", clade_name)
  clade_token <- tryCatch({
    parts <- strsplit(clade_name_norm, "\\.")[[1]]
    tail(parts, 1)
  }, error = function(e) clade_name_norm)
  
  # Try different matching strategies
  clade_idx <- which(
    node_data$name == clade_name |
    node_data$name_norm == clade_name_norm |
    node_data$hgca_celltype_level1 == clade_name |
    node_data$hgca_celltype_level1_majority == clade_name |
    node_data$hgca_celltype_v1_majority == clade_name |
    node_data$label_clean == clade_name |
    node_data$label_clean == clade_token
  )
  
  if (length(clade_idx) == 0) {
    # Fallback: try token-based hgca_celltype_level1 subsetting (e.g., "Lymphoid")
    token <- clade_token
    cand <- which(
      node_data$hgca_celltype_level1 == token |
      node_data$hgca_celltype_level1_majority == token |
      node_data$label_clean == token |
      grepl(paste0("[\\.\\|]", token, "$"), node_data$name) |
      grepl(paste0("[\\.\\|]", token, "$"), node_data$name_norm)
    )
    if (length(cand) == 0) {
      cat("⚠️ Clade not found:", clade_name, "\n")
      return(graph)
    }
    clade_idx <- cand[1]
  }
  
  clade_idx <- clade_idx[1]
  parent_name <- node_data$name[clade_idx]
  parent_name_norm <- node_data$name_norm[clade_idx]
  
  # Find all descendants using name hierarchy
  descendants <- which(startsWith(node_data$name, parent_name) | startsWith(node_data$name_norm, parent_name_norm))
  
  # Subset the graph
  subgraph <- graph %N>% filter(row_number() %in% descendants)
  
  cat("✅ Subset created\n")
  
  return(subgraph)
}

# Function to calculate optimal plot size
calculate_plot_size <- function(n_nodes) {
  # Original full tree: ~100 nodes at 14x18 inches
  # Scale proportionally with sqrt for diminishing returns
  
  min_width <- 10
  min_height <- 8
  
  scale_factor <- sqrt(n_nodes / 100)
  
  width <- max(min_width, 14 * scale_factor)
  height <- max(min_height, 18 * scale_factor)
  
  # Cap at reasonable maximum
  width <- min(width, 20)
  height <- min(height, 30)
  
  return(list(width = width, height = height))
}

# Helper function to normalize cell type labels for robust matching
normalize_label_for_taxonomy <- function(label) {
  label <- as.character(label)
  # Fix "vascular endothelial" -> "Endothelial"
  label <- stringr::str_replace(label, "^vascular endothelial", "Endothelial")
  # Fix "ILC NK Cells" -> "NK Cells" (taxonomy uses "NK Cells" not "ILC NK Cells")
  label <- stringr::str_replace(label, "^ILC NK Cells$", "NK Cells")
  # Fix "ILC NKs" -> "NK Cells" (variant)
  label <- stringr::str_replace(label, "^ILC NKs$", "NK Cells")
  # Could add more mappings here as needed
  return(label)
}

# Function to load and calculate per-class accuracy
calculate_perclass_accuracy <- function(predictions_csv) {
  # Load predictions CSV
  if (!file.exists(predictions_csv)) {
    cat("⚠️ Predictions not found:", predictions_csv, "\n")
    return(NULL)
  }
  
  preds <- read_csv(predictions_csv, show_col_types = FALSE)
  
  # Normalize labels to match taxonomy
  preds <- preds %>%
    mutate(
      true_label = normalize_label_for_taxonomy(true_label),
      predicted_label = normalize_label_for_taxonomy(predicted_label)
    )
  
  # Calculate accuracy per cell type
  perclass_acc <- preds %>%
    group_by(true_label) %>%
    summarise(
      n_total = n(),
      n_correct = sum(true_label == predicted_label),
      accuracy = n_correct / n_total,
      .groups = "drop"
    ) %>%
    rename(cell_type = true_label)
  
  return(perclass_acc)
}

# Function to calculate usage from predictions (leaf-level, with full support)
calculate_usage_from_predictions <- function(predictions_csv, leaf_names = NULL) {
  # Calculate what cell types were predicted (not ground truth)

  if (!file.exists(predictions_csv)) {
    cat("⚠️ Predictions not found:", predictions_csv, "\n")
    return(NULL)
  }

  preds <- read_csv(predictions_csv, show_col_types = FALSE)
  
  # Normalize labels to match taxonomy (e.g., "vascular endothelial" -> "Endothelial")
  preds <- preds %>%
    mutate(predicted_label = normalize_label_for_taxonomy(predicted_label))

  # Normalize helper for case-insensitive joins
  norm <- function(x) {
    x <- as.character(x)
    x <- stringr::str_squish(x)
    x <- tolower(x)
    x
  }

  # Counts for labels present in predictions (normalized)
  present_counts <- preds %>%
    transmute(cell_type = predicted_label) %>%
    filter(!is.na(cell_type), !cell_type %in% c("doublet", "lowQ", "unclear", "unknown")) %>%
    mutate(cell_type_norm = norm(cell_type)) %>%
    count(cell_type_norm, name = "n_cells")

  # Expand to full leaf set if provided (fill missing with zeros)
  if (!is.null(leaf_names)) {
    all_leaves <- tibble(cell_type = as.character(leaf_names)) %>%
      mutate(cell_type_norm = norm(cell_type))
    label_counts <- all_leaves %>%
      left_join(present_counts, by = "cell_type_norm") %>%
      mutate(n_cells = if_else(is.na(n_cells), 0L, n_cells)) %>%
      select(cell_type, n_cells)
  } else {
    # If no leaf set given, return normalized keys as original
    label_counts <- present_counts %>%
      rename(cell_type = cell_type_norm)
  }

  total_cells <- sum(label_counts$n_cells)
  label_counts <- label_counts %>%
    mutate(usage_pct = ifelse(total_cells > 0, (n_cells / total_cells) * 100, 0))

  return(label_counts)
}

# Propagate leaf metrics (accuracy, usage) up the taxonomy to internal nodes
propagate_metrics_to_nodes <- function(graph, perclass_leaves, usage_leaves) {
  node_df <- graph %N>% as_tibble()
  
  # Normalize helper for robust joins
  norm <- function(x) {
    x <- as.character(x)
    x <- stringr::str_squish(x)
    x <- tolower(x)
    x
  }

  # Normalize keys in both maps
  perclass_map <- perclass_leaves %>% 
    select(cell_type, n_total, n_correct, accuracy) %>%
    mutate(cell_type_norm = norm(cell_type))
  usage_map <- usage_leaves %>% 
    select(cell_type, n_cells, usage_pct) %>%
    mutate(cell_type_norm = norm(cell_type))

  total_preds <- sum(usage_map$n_cells, na.rm = TRUE)

  aggregate_one <- function(node_name) {
    descendant_idx <- which(startsWith(node_df$name, node_name))
    descendant_leaves <- node_df$hgca_celltype_v1_majority[descendant_idx]
    descendant_leaves <- descendant_leaves[!is.na(descendant_leaves)]
    descendant_leaves_norm <- norm(descendant_leaves)

    pc <- perclass_map %>% filter(cell_type_norm %in% descendant_leaves_norm)
    n_total_sum <- sum(pc$n_total, na.rm = TRUE)
    n_correct_sum <- sum(pc$n_correct, na.rm = TRUE)
    acc <- ifelse(n_total_sum > 0, n_correct_sum / n_total_sum, NA_real_)

    us <- usage_map %>% filter(cell_type_norm %in% descendant_leaves_norm)
    n_cells_sum <- sum(us$n_cells, na.rm = TRUE)
    usage <- ifelse(total_preds > 0, (n_cells_sum / total_preds) * 100, 0)

    tibble(method_accuracy = acc, usage_pct = usage)
  }

  agg <- map_dfr(node_df$name, aggregate_one)

  graph %N>% mutate(
    method_accuracy = agg$method_accuracy,
    usage_pct = agg$usage_pct
  )
}

# Function to create ARBOL dendrogram (sideways style)
create_arbol_sideways <- function(graph_with_metrics, 
                                  title,
                                  color_var = "method_accuracy",
                                  color_label = "Accuracy",
                                  show_internal_labels = TRUE,
                                  layout_type = "dendrogram") {  # "dendrogram" or "tree"
  # Create sideways ARBOL plot matching your preferred style
  # From baseline: options(repr.plot.width=14,repr.plot.height=18) + coord_flip()
  
  # Calculate optimal size (get node count from tidygraph)
  n_nodes <- graph_with_metrics %N>% as_tibble() %>% nrow()
  plot_size <- calculate_plot_size(n_nodes)
  
  cat(sprintf("   📐 Plot size for %d nodes: %.1f × %.1f inches (%s layout)\n", 
              n_nodes, plot_size$width, plot_size$height, layout_type))
  
  # Set up color scale
  if (color_var == "method_accuracy") {
    color_scale <- scale_edge_colour_gradient2(
      low = "red", mid = "yellow", high = "darkgreen",
      midpoint = 0.5, limits = c(0, 1),
      name = color_label,
      guide = guide_colourbar(barwidth = 1, barheight = 10)
    )
  } else if (color_var == "usage_pct") {
    color_scale <- scale_edge_colour_viridis(
      option = "plasma",
      name = color_label
    )
  } else {
    color_scale <- scale_edge_colour_gradient(low = "lightblue", high = "darkblue")
  }
  
  # Wes Anderson palette for hgca_celltype_level1
  library(wesanderson)
  cols <- wes_palette('Zissou1')[c(1, 5, 4, 6)]
  names(cols) <- c("Immune", "Stromal", "Epithelial", "Neurons")
  edge_cols <- c(cols, "root" = "black")
  
  # Create plot with chosen layout
  if (layout_type == "dendrogram") {
    # Dendrogram layout - need to manually add color attribute to edges
    node_data <- graph_with_metrics %N>% as_tibble() %>% mutate(.tidygraph_node_index = row_number())
    edge_data <- graph_with_metrics %E>% as_tibble()
    
    # Join color from 'to' node (using integer index)
    edge_data <- edge_data %>%
      left_join(
        node_data %>% select(.tidygraph_node_index, edge_color_val = !!sym(color_var)), 
        by = c("to" = ".tidygraph_node_index")
      )
    
    # Add edge color attribute
    graph_with_edge_colors <- graph_with_metrics %E>%
      mutate(edge_color_val = edge_data$edge_color_val)
    
    p <- ggraph(graph_with_edge_colors, layout = 'dendrogram') +
      geom_edge_diagonal(aes(color = edge_color_val), linewidth = 1.5) +
      color_scale
  } else {
    # tree layout (original) - diagonal2 supports node.* syntax
    p <- ggraph(graph_with_metrics, layout = 'tree') +
      geom_edge_diagonal2(aes_string(color = paste0("node.", color_var)), linewidth = 1.5) +
      color_scale
  }
  
  # Add node labels (leaves)
  if (show_internal_labels) {
    p <- p +
      geom_node_text(
        aes(filter = numChildren == 0, 
            label = hgca_celltype_v1_majority, 
            color = hgca_celltype_level1_majority),
        nudge_y = 0.1, vjust = 0.5, hjust = 0, size = 5
      ) +
      geom_node_text(
        aes(filter = numChildren > 0, 
            label = ifelse(grepl("\\|", name), sub('.*\\|','', name), sub('.*\\.', '', name))),
        color = 'black', size = 5, repel = TRUE
      )
  } else {
    # Minimal version - only leaf labels
    p <- p +
      geom_node_text(
        aes(filter = numChildren == 0, 
            label = hgca_celltype_v1_majority, 
            color = hgca_celltype_level1_majority),
        nudge_y = 0.1, vjust = 0.5, hjust = 0, size = 5
      )
  }
  
  # Styling
  p <- p +
    theme_linedraw(base_size = 16) +
    scale_colour_manual(values = cols) +
    guides(
      edge_colour = guide_legend(title = color_label),
      color = guide_legend(title = 'hgca_celltype_level1')
    ) +
    ggtitle(title) +
    scale_y_reverse(
      breaks = seq(0, 5, by = 1),
      labels = c("Whim", "Status", "Type", "Differentia", "hgca_celltype_level1", "Root")
    ) +
    theme(
      axis.text.y = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    labs(
      title = title,
      x = "Taxa",
      y = "Classification Ranks"
    ) +
    coord_flip() +
    expand_limits(y = -5)
  
  return(list(plot = p, width = plot_size$width, height = plot_size$height))
}

# Build full taxonomy graph from CSV (dot-path nodes)
build_taxonomy_graph_from_csv <- function(taxonomy_csv) {
  tax <- suppressMessages(readr::read_csv(taxonomy_csv, show_col_types = FALSE))

  # Columns in hierarchical order
  hier_cols <- c("hgca_celltype_level1", "Differentia", "Habitus", "Status", "Whim", "hgca_celltype_v1")
  hier_cols <- hier_cols[hier_cols %in% colnames(tax)]

  # Build cumulative dot-path per row
  make_path_nodes <- function(row) {
    parts <- c("root", as.character(row[hier_cols]))
    parts <- parts[!is.na(parts) & parts != ""]
    nodes <- Reduce(function(acc, x) c(acc, paste0(tail(acc, 1), ".", x)), parts[-1], init = parts[1])
    # nodes now: root, root.hgca_celltype_level1, root.hgca_celltype_level1.Differentia, ... , leaf
    nodes
  }

  paths <- apply(tax[, hier_cols, drop = FALSE], 1, make_path_nodes)

  # Build edges between consecutive nodes for all paths
  edge_list <- map(paths, function(nodes) tibble(from = head(nodes, -1), to = tail(nodes, 1)))
  edges <- bind_rows(edge_list) %>% distinct()

  # Nodes table
  node_names <- unique(c(edges$from, edges$to))
  nodes <- tibble(name = node_names)

  # Derive attributes
  # Derive hgca_celltype_level1 using second token of dot-path (root.hgca_celltype_level1....)
  hgca_celltype_level1_vec <- map_chr(str_split(nodes$name, fixed(".")), function(x) if (length(x) >= 2) x[2] else NA_character_)
  nodes <- nodes %>% mutate(
    # last token after last dot for label display of internal nodes
    label = sub('.*\\.', '', name),
    hgca_celltype_level1 = if_else(name == "root", "root", hgca_celltype_level1_vec),
    hgca_celltype_level1 = if_else(is.na(hgca_celltype_level1), "root", hgca_celltype_level1),
    hgca_celltype_level1_majority = hgca_celltype_level1
  )

  # Mark leaf nodes with hgca_celltype_v1_majority by matching last step in path
  # Build lookup of leaf path -> leaf label
  leaf_lookup <- map_dfr(paths, function(nodes) tibble(path = tail(nodes, 1), leaf = sub('.*\\.', '', tail(nodes, 1)))) %>% distinct()
  nodes <- nodes %>% left_join(leaf_lookup, by = c("name" = "path")) %>%
    rename(hgca_celltype_v1_majority = leaf)

  # Compute numChildren from edge list (string-based, before graph conversion)
  edge_counts <- edges %>% count(from, name = "numChildren")
  nodes <- nodes %>%
    left_join(edge_counts, by = c("name" = "from")) %>%
    mutate(numChildren = if_else(is.na(numChildren), 0L, as.integer(numChildren)))

  # Build graph
  tbl_graph(nodes = nodes, edges = edges, directed = TRUE)
}

# Main function to create ARBOL plots for a hgca_celltype_level1
create_hgca_celltype_level1_arbol_plots <- function(hgca_celltype_level1_config_path,
                                       results_dir,
                                       methods = c("scanvi", "scimilarity_base", "scimilarity_lora"),
                                       label_types = c("hgca_celltype_v1", "hgca_celltype_level_1")) {
  # Complete pipeline to create all ARBOL plots for a hgca_celltype_level1
  
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat("CREATING ARBOL PLOTS FOR hgca_celltype_level1\n")
  cat(strrep("=", 70), "\n\n")
  
  # Load config
  library(yaml)
  config <- read_yaml(hgca_celltype_level1_config_path)
  
  hgca_celltype_level1 <- config$hgca_celltype_level1
  cat("hgca_celltype_level1:", config$display_name, "\n")
  cat("Results dir:", results_dir, "\n\n")
  
  # Load taxonomy graph (prefer the curated RDS layout to avoid hairball)
  cat("Loading ARBOL taxonomy...\n")
  if (!is.null(config$taxonomy$arbol_graph) && file.exists(config$taxonomy$arbol_graph)) {
    tax_graph <- readRDS(config$taxonomy$arbol_graph)
    cat("✅ Loaded taxonomy graph from RDS\n")
  } else if (!is.null(config$taxonomy$gca_taxonomy_path) && file.exists(config$taxonomy$gca_taxonomy_path)) {
    tax_graph <- build_taxonomy_graph_from_csv(config$taxonomy$gca_taxonomy_path)
    cat("✅ Built taxonomy graph from CSV\n")
  } else {
    stop("No valid taxonomy source found in config")
  }

  # Normalize/ensure required fields exist regardless of source
  tax_graph <- ensure_required_node_fields(tax_graph)
  
  # Subset to hgca_celltype_level1 if specified
  if (!is.null(config$taxonomy$arbol_subset)) {
    cat("Subsetting to:", config$taxonomy$arbol_subset, "\n")
    tax_graph <- subset_to_clade(tax_graph, config$taxonomy$arbol_subset)
    # Re-ensure fields after subsetting
    tax_graph <- ensure_required_node_fields(tax_graph)
  }
  
  # Create plots for each method and label type
  for (method in methods) {
    for (label_type in label_types) {
      cat("\n--- ", method, "-", label_type, "---\n")
      
      # Load predictions
      pred_file <- file.path(results_dir, "predictions", 
                            paste0("predictions_", method, "_", label_type, ".csv"))
      
      if (!file.exists(pred_file)) {
        cat("⚠️ Predictions not found, skipping\n")
        next
      }
      
      # Determine leaf names in current subset (for full-support joins)
      leaf_names <- (tax_graph %N>% as_tibble()) %>%
        filter(numChildren == 0) %>%
        pull(hgca_celltype_v1_majority) %>% as.character()

      # Leaf-level metrics
      perclass_acc <- calculate_perclass_accuracy(pred_file)
      if (is.null(perclass_acc)) next
      usage <- calculate_usage_from_predictions(pred_file, leaf_names = leaf_names)
      if (is.null(usage)) next

      # Propagate leaf metrics up to all nodes (full taxonomy for the subset)
      graph_with_metrics <- propagate_metrics_to_nodes(tax_graph, perclass_acc, usage)
      
      # Create accuracy plot
      result_acc <- create_arbol_sideways(
        graph_with_metrics,
        paste0(method, " - ", label_type, " (Accuracy)"),
        color_var = "method_accuracy",
        color_label = "Accuracy",
        show_internal_labels = TRUE
      )
      
      # Save accuracy plot
      filename_acc <- file.path(results_dir, "plots",
                               paste0("arbol_", method, "_", label_type, "_accuracy.png"))
      ggsave(filename_acc, result_acc$plot, 
             width = result_acc$width, height = result_acc$height, dpi = 300)
      cat("✅ Saved:", basename(filename_acc), "\n")
      
      # Create usage plot
      result_usage <- create_arbol_sideways(
        graph_with_metrics,
        paste0(method, " - ", label_type, " (Usage %)"),
        color_var = "usage_pct",
        color_label = "Usage %",
        show_internal_labels = TRUE
      )
      
      # Save usage plot
      filename_usage <- file.path(results_dir, "plots",
                                 paste0("arbol_", method, "_", label_type, "_usage.png"))
      ggsave(filename_usage, result_usage$plot,
             width = result_usage$width, height = result_usage$height, dpi = 300)
      cat("✅ Saved:", basename(filename_usage), "\n")
    }
  }
  
  cat("\n✅ All ARBOL plots created!\n")
}

# Function to create ARBOL presence plot (binary: present vs absent)
create_arbol_presence <- function(graph_with_presence,
                                   title,
                                   show_internal_labels = TRUE,
                                   show_cell_counts = TRUE) {
  # Create ARBOL showing binary presence (grey for absent, bright for present)
  # Only leaves and entirely unused clades are greyed out
  # Optionally shows cell counts next to leaf labels
  
  n_nodes <- graph_with_presence %N>% as_tibble() %>% nrow()
  plot_size <- calculate_plot_size(n_nodes)
  
  # Add extra width for cell counts if showing them
  if (show_cell_counts) {
    plot_size$width <- plot_size$width * 1.2  # 20% wider for count labels
  }
  
  cat(sprintf("   📐 Plot size for %d nodes: %.1f × %.1f inches (presence plot)\n", 
              n_nodes, plot_size$width, plot_size$height))
  
  # Wes Anderson palette
  library(wesanderson)
  cols <- wes_palette('Zissou1')[c(1, 5, 4, 6)]
  names(cols) <- c("Immune", "Stromal", "Epithelial", "Neurons")
  
  # Create plot - use dendrogram layout with manual edge color assignment
  node_data <- graph_with_presence %N>% as_tibble() %>% mutate(.tidygraph_node_index = row_number())
  edge_data <- graph_with_presence %E>% as_tibble()
  
  # Join presence from 'to' node
  edge_data <- edge_data %>%
    left_join(
      node_data %>% select(.tidygraph_node_index, edge_presence = is_present), 
      by = c("to" = ".tidygraph_node_index")
    )
  
  # Add to graph
  graph_with_edge_presence <- graph_with_presence %E>%
    mutate(edge_presence = edge_data$edge_presence)
  
  # Determine which values are present in the data
  edge_presence_vals <- graph_with_edge_presence %E>% pull(edge_presence) %>% unique() %>% sort()
  
  # Create color scale based on what values exist
  if (length(edge_presence_vals) == 1) {
    # Only one value present - use discrete scale
    if (edge_presence_vals[1] == 0) {
      # All absent
      p <- ggraph(graph_with_edge_presence, layout = 'dendrogram') +
        geom_edge_diagonal(aes(color = edge_presence), linewidth = 1.5) +
        scale_edge_colour_gradient(
          low = "grey80", high = "grey80",
          name = "Present",
          limits = c(0, 1),
          breaks = c(0),
          labels = c("Absent")
        )
    } else {
      # All present
      p <- ggraph(graph_with_edge_presence, layout = 'dendrogram') +
        geom_edge_diagonal(aes(color = edge_presence), linewidth = 1.5) +
        scale_edge_colour_gradient(
          low = "#FF6B35", high = "#FF6B35",
          name = "Present",
          limits = c(0, 1),
          breaks = c(1),
          labels = c("Present")
        )
    }
  } else {
    # Both present and absent - use full gradient
    p <- ggraph(graph_with_edge_presence, layout = 'dendrogram') +
      geom_edge_diagonal(aes(color = edge_presence), linewidth = 1.5) +
      scale_edge_colour_gradient(
        low = "grey80",    # Absent - light grey
        high = "#FF6B35",  # Present - bright coral/orange
        name = "Present",
        limits = c(0, 1),
        breaks = c(0, 1),
        labels = c("Absent", "Present")
      )
  }
  
  # Add node labels
  if (show_internal_labels) {
    p <- p +
      geom_node_text(
        aes(filter = numChildren == 0, 
            label = hgca_celltype_v1_majority,
            alpha = is_present),  # Dim absent labels
        nudge_y = 0.1, vjust = 0.5, hjust = 0, size = 5, color = "black"
      ) +
      geom_node_text(
        aes(filter = numChildren > 0, 
            label = ifelse(grepl("\\|", name), sub('.*\\|','', name), sub('.*\\.', '', name)),
            alpha = is_present),  # Dim absent internal nodes
        color = 'black', size = 5, repel = TRUE
      ) +
      scale_alpha_continuous(range = c(0.2, 1.0), guide = "none")  # Absent=20%, present=100%
    
    # Add cell counts for present leaves
    if (show_cell_counts) {
      p <- p +
        geom_node_text(
          aes(filter = numChildren == 0 & is_present > 0,
              label = paste0("n=", scales::comma(n_cells_leaf))),
          nudge_y = 0.15, nudge_x = 0.5, vjust = 0.5, hjust = 0,
          size = 3.5, color = "grey40", fontface = "italic"
        )
    }
  } else {
    p <- p +
      geom_node_text(
        aes(filter = numChildren == 0, 
            label = hgca_celltype_v1_majority,
            alpha = is_present),
        nudge_y = 0.1, vjust = 0.5, hjust = 0, size = 5, color = "black"
      ) +
      scale_alpha_continuous(range = c(0.2, 1.0), guide = "none")
    
    if (show_cell_counts) {
      p <- p +
        geom_node_text(
          aes(filter = numChildren == 0 & is_present > 0,
              label = paste0("n=", scales::comma(n_cells_leaf))),
          nudge_y = 0.15, nudge_x = 0.5, vjust = 0.5, hjust = 0,
          size = 3.5, color = "grey40", fontface = "italic"
        )
    }
  }
  
  # Styling
  p <- p +
    theme_linedraw(base_size = 16) +
    ggtitle(title) +
    scale_y_reverse(
      breaks = seq(0, 5, by = 1),
      labels = c("Whim", "Status", "Type", "Differentia", "hgca_celltype_level1", "Root")
    ) +
    theme(
      axis.text.y = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    labs(
      title = title,
      x = "Taxa",
      y = "Classification Ranks"
    ) +
    coord_flip() +
    expand_limits(y = -5)
  
  return(list(plot = p, width = plot_size$width, height = plot_size$height))
}

# Function to create before/after usage comparison ARBOL plots
create_usage_comparison_arbol <- function(graph,
                                          original_labels_vec,
                                          transferred_labels_vec,
                                          hgca_celltype_level1_name,
                                          output_dir,
                                          subset_clade = NULL) {
  # Create side-by-side ARBOL plots showing taxonomy coverage before/after label transfer
  # 
  # Args:
  #   graph: tbl_graph with full GCA taxonomy
  #   original_labels_vec: named vector of cell counts for original annotations
  #   transferred_labels_vec: named vector of cell counts for transferred GCA labels
  #   hgca_celltype_level1_name: Name of hgca_celltype_level1 (e.g., "Myeloid", "Lymphoid")
  #   output_dir: Directory to save plots
  #   subset_clade: Optional clade name to subset (e.g., "root.Immune.Lymphoid")
  
  cat("\n", strrep("=", 70), "\n")
  cat("ARBOL: USAGE COMPARISON -", hgca_celltype_level1_name, "\n")
  cat(strrep("=", 70), "\n")
  
  # Subset if requested
  if (!is.null(subset_clade)) {
    cat("📐 Subsetting to clade:", subset_clade, "\n")
    graph <- subset_to_clade(graph, subset_clade)
  }
  
  # Get leaf names from graph
  node_data <- graph %N>% as_tibble()
  leaf_names <- node_data %>%
    filter(numChildren == 0) %>%
    pull(hgca_celltype_v1_majority) %>%
    na.omit() %>%
    unique()
  
  cat("📊 Taxonomy leaves:", length(leaf_names), "\n")
  cat("📊 Original labels:", length(original_labels_vec), "\n")
  cat("📊 Transferred labels:", length(transferred_labels_vec), "\n")
  
  # Normalize label names for matching
  norm <- function(x) {
    x <- as.character(x)
    x <- normalize_label_for_taxonomy(x)
    # Remove newlines and extra whitespace
    x <- gsub("\\n", " ", x)  # Replace newlines with spaces
    x <- gsub("\\r", " ", x)  # Replace carriage returns with spaces
    x <- stringr::str_squish(x)  # Collapse multiple spaces
    x <- tolower(x)
    x
  }
  
  # Map original labels to leaf counts
  original_df <- tibble(
    label = names(original_labels_vec),
    count = as.integer(original_labels_vec)
  ) %>%
    mutate(label_norm = norm(label))
  
  # Map transferred labels to leaf counts
  transferred_df <- tibble(
    label = names(transferred_labels_vec),
    count = as.integer(transferred_labels_vec)
  ) %>%
    mutate(label_norm = norm(label))
  
  # Create leaf usage tibbles (matching taxonomy leaf names)
  leaves_df <- tibble(
    cell_type = as.character(leaf_names),
    cell_type_norm = norm(cell_type)
  )
  
  # DEBUG: Print sample labels for matching
  cat("🔍 DEBUG - Label matching:\n")
  cat("   Leaf names (first 5):", paste(head(leaf_names, 5), collapse = ", "), "\n")
  cat("   Leaf normalized (first 5):", paste(head(leaves_df$cell_type_norm, 5), collapse = ", "), "\n")
  cat("   Transferred labels (first 5):", paste(head(names(transferred_labels_vec), 5), collapse = ", "), "\n")
  cat("   Transferred normalized (first 5):", paste(head(transferred_df$label_norm, 5), collapse = ", "), "\n")
  
  original_usage <- leaves_df %>%
    left_join(original_df, by = c("cell_type_norm" = "label_norm")) %>%
    mutate(n_cells = if_else(is.na(count), 0L, count)) %>%
    select(cell_type, n_cells)
  
  transferred_usage <- leaves_df %>%
    left_join(transferred_df, by = c("cell_type_norm" = "label_norm")) %>%
    mutate(n_cells = if_else(is.na(count), 0L, count)) %>%
    select(cell_type, n_cells)
  
  total_original <- sum(original_usage$n_cells)
  total_transferred <- sum(transferred_usage$n_cells)
  
  # DEBUG: Show matches
  cat("🔍 DEBUG - Matches found:\n")
  cat("   Original matched:", sum(original_usage$n_cells > 0), "cell types\n")
  cat("   Transferred matched:", sum(transferred_usage$n_cells > 0), "cell types\n")
  if (sum(transferred_usage$n_cells > 0) == 0) {
    cat("   ⚠️  No transferred matches! Checking for any partial matches...\n")
    # Try to find why no matches
    for (i in 1:min(5, nrow(transferred_df))) {
      trans_label <- transferred_df$label_norm[i]
      matches <- leaves_df$cell_type_norm[grepl(trans_label, leaves_df$cell_type_norm, fixed = TRUE)]
      if (length(matches) > 0) {
        cat("      '", transferred_df$label[i], "' (norm: '", trans_label, "') could match: ", paste(matches, collapse = ", "), "\n", sep = "")
      }
    }
  }
  
  original_usage <- original_usage %>%
    mutate(usage_pct = ifelse(total_original > 0, (n_cells / total_original) * 100, 0))
  
  transferred_usage <- transferred_usage %>%
    mutate(usage_pct = ifelse(total_transferred > 0, (n_cells / total_transferred) * 100, 0))
  
  cat("✅ Original: ", sum(original_usage$n_cells > 0), " cell types used\n")
  cat("✅ Transferred: ", sum(transferred_usage$n_cells > 0), " cell types used\n")
  
  # Propagate to nodes (using dummy accuracy = NA)
  perclass_dummy <- tibble(cell_type = character(), n_total = integer(), 
                           n_correct = integer(), accuracy = numeric())
  
  graph_original <- propagate_metrics_to_nodes(graph, perclass_dummy, original_usage)
  graph_transferred <- propagate_metrics_to_nodes(graph, perclass_dummy, transferred_usage)
  
  # Create plots - BOTH tree and dendrogram layouts
  cat("\n🎨 Creating tree layout plots...\n")
  plot_original_tree <- create_arbol_sideways(
    graph_original,
    title = paste0(hgca_celltype_level1_name, ": Original Annotations (Tree)"),
    color_var = "usage_pct",
    color_label = "Usage %",
    show_internal_labels = TRUE,
    layout_type = "tree"
  )
  
  plot_transferred_tree <- create_arbol_sideways(
    graph_transferred,
    title = paste0(hgca_celltype_level1_name, ": After GCA Transfer (Tree)"),
    color_var = "usage_pct",
    color_label = "Usage %",
    show_internal_labels = TRUE,
    layout_type = "tree"
  )
  
  cat("🎨 Creating dendrogram layout plots...\n")
  plot_original_dendro <- create_arbol_sideways(
    graph_original,
    title = paste0(hgca_celltype_level1_name, ": Original Annotations (Dendrogram)"),
    color_var = "usage_pct",
    color_label = "Usage %",
    show_internal_labels = TRUE,
    layout_type = "dendrogram"
  )
  
  plot_transferred_dendro <- create_arbol_sideways(
    graph_transferred,
    title = paste0(hgca_celltype_level1_name, ": After GCA Transfer (Dendrogram)"),
    color_var = "usage_pct",
    color_label = "Usage %",
    show_internal_labels = TRUE,
    layout_type = "dendrogram"
  )
  
  # Save plots
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Tree layout files
  fname_orig_tree <- file.path(output_dir, paste0(tolower(hgca_celltype_level1_name), "_original_tree.png"))
  fname_trans_tree <- file.path(output_dir, paste0(tolower(hgca_celltype_level1_name), "_transferred_tree.png"))
  
  # Dendrogram layout files
  fname_orig_dendro <- file.path(output_dir, paste0(tolower(hgca_celltype_level1_name), "_original_dendrogram.png"))
  fname_trans_dendro <- file.path(output_dir, paste0(tolower(hgca_celltype_level1_name), "_transferred_dendrogram.png"))
  
  ggsave(fname_orig_tree, plot_original_tree$plot, 
         width = plot_original_tree$width, height = plot_original_tree$height, dpi = 300)
  ggsave(fname_trans_tree, plot_transferred_tree$plot, 
         width = plot_transferred_tree$width, height = plot_transferred_tree$height, dpi = 300)
  ggsave(fname_orig_dendro, plot_original_dendro$plot, 
         width = plot_original_dendro$width, height = plot_original_dendro$height, dpi = 300)
  ggsave(fname_trans_dendro, plot_transferred_dendro$plot, 
         width = plot_transferred_dendro$width, height = plot_transferred_dendro$height, dpi = 300)
  
  # Create PRESENCE plots (binary: present vs absent)
  cat("🎨 Creating presence/absence plots (grey vs bright)...\n")
  
  # Add binary presence flag and actual cell counts to graphs
  # Calculate actual counts from usage_pct
  total_original <- sum(original_usage$n_cells)
  total_transferred <- sum(transferred_usage$n_cells)
  
  graph_original_binary <- graph_original %N>% 
    mutate(
      is_present = ifelse(usage_pct > 0, 1, 0),
      n_cells_leaf = ifelse(numChildren == 0, round(usage_pct / 100 * total_original), 0)
    )
  
  graph_transferred_binary <- graph_transferred %N>% 
    mutate(
      is_present = ifelse(usage_pct > 0, 1, 0),
      n_cells_leaf = ifelse(numChildren == 0, round(usage_pct / 100 * total_transferred), 0)
    )
  
  # Create presence plots with custom colors (grey vs bright)
  plot_original_presence <- create_arbol_presence(
    graph_original_binary,
    title = paste0(hgca_celltype_level1_name, ": Original Annotations (Presence)"),
    show_internal_labels = TRUE
  )
  
  plot_transferred_presence <- create_arbol_presence(
    graph_transferred_binary,
    title = paste0(hgca_celltype_level1_name, ": After GCA Transfer (Presence)"),
    show_internal_labels = TRUE
  )
  
  # Save all plots
  fname_orig_presence <- file.path(output_dir, paste0(tolower(hgca_celltype_level1_name), "_original_presence.png"))
  fname_trans_presence <- file.path(output_dir, paste0(tolower(hgca_celltype_level1_name), "_transferred_presence.png"))
  
  ggsave(fname_orig_presence, plot_original_presence$plot, 
         width = plot_original_presence$width, height = plot_original_presence$height, dpi = 300)
  ggsave(fname_trans_presence, plot_transferred_presence$plot, 
         width = plot_transferred_presence$width, height = plot_transferred_presence$height, dpi = 300)
  
  cat("\n💾 Saved (6 files):\n")
  cat("   Tree layout:\n")
  cat("     - Original:", fname_orig_tree, "\n")
  cat("     - Transferred:", fname_trans_tree, "\n")
  cat("   Dendrogram layout (more readable):\n")
  cat("     - Original:", fname_orig_dendro, "\n")
  cat("     - Transferred:", fname_trans_dendro, "\n")
  cat("   Presence/Absence (grey vs bright):\n")
  cat("     - Original:", fname_orig_presence, "\n")
  cat("     - Transferred:", fname_trans_presence, "\n")
  
  return(list(
    original_tree = plot_original_tree,
    transferred_tree = plot_transferred_tree,
    original_dendro = plot_original_dendro,
    transferred_dendro = plot_transferred_dendro,
    original_presence = plot_original_presence,
    transferred_presence = plot_transferred_presence,
    original_usage = original_usage,
    transferred_usage = transferred_usage
  ))
}

# Manual overrides for taxonomy nodes that are stored as concatenated
# lowercase strings (e.g. "Folliclassociatedenterocyte FAE" or
# "Subepithelialfibroblasts S2"). These cannot be reliably split by regex
# alone because the word boundaries aren't marked by case changes; without
# a dictionary, "subepithelialfibroblasts" can't be recovered as
# "subepithelial fibroblasts" automatically.
TAXONOMY_NODE_OVERRIDES <- c(
  "SubepithelialfibroblastsS2"      = "Subepithelial Fibroblasts (S2)",
  "FibroblasticreticularcellsFRC"   = "Fibroblastic Reticular Cells (FRC)",
  "FollicleassociatedenterocyteFAE" = "Follicle Associated Enterocyte (FAE)",
  "TransientlyAmplifyingCellsTA"    = "Transiently Amplifying Cells (TA)",
  "EnteroendocrineCellsEEC"         = "Enteroendocrine Cells (EEC)",
  "InnateLymphoidCellsILC"          = "Innate Lymphoid Cells (ILC)",
  "GerminalCenterBCellsGCB"         = "Germinal Center B Cells (GCB)"
)

# Insert spaces into CamelCase / mixed-case identifiers so that the dot-path
# tokens used as taxonomy node names (e.g. "InnateLymphoidCellsILC",
# "GerminalCenterBCellsGCB", "CD4TCells") render as their original cell-type
# names (e.g. "Innate Lymphoid Cells (ILC)", "Germinal Center B Cells (GCB)",
# "CD4 T Cells"). Falls back to TAXONOMY_NODE_OVERRIDES for known
# all-lowercase joined names.
un_camel_case <- function(x) {
  x <- as.character(x)
  # 1) explicit overrides for known concatenated lowercase taxonomy nodes
  override_idx <- match(x, names(TAXONOMY_NODE_OVERRIDES))
  has_override <- !is.na(override_idx)
  out <- x
  out[has_override] <- TAXONOMY_NODE_OVERRIDES[override_idx[has_override]]
  # 2) regex-based CamelCase splitting for everything else
  if (any(!has_override)) {
    y <- out[!has_override]
    y <- gsub("([0-9])([A-Z])",       "\\1 \\2", y, perl = TRUE)
    y <- gsub("([a-z])([A-Z])",       "\\1 \\2", y, perl = TRUE)
    y <- gsub("([A-Z])([A-Z][a-z])",  "\\1 \\2", y, perl = TRUE)
    # Wrap a trailing all-uppercase token in parentheses, e.g.
    # "Innate Lymphoid Cells ILC" -> "Innate Lymphoid Cells (ILC)".
    y <- sub("\\s+([A-Z]{2,})$", " (\\1)", y, perl = TRUE)
    out[!has_override] <- y
  }
  stringr::str_squish(out)
}

# Helper: inject extra leaves under named parent nodes of a tbl_graph.
#
# `extras` must be a data.frame / tibble with at least:
#   - parent_path : exact `name` of an existing node (e.g.
#                   "Epithelial.SecretoryEpithelial.TuftCells")
#   - label       : the new leaf's displayed cell-type label (used both as
#                   `hgca_celltype_v1_majority` for matching against
#                   *_labels_vec and rendered on the plot)
#
# The new node's `name` is `paste0(parent_path, ".", safe_token)` where the
# safe token strips whitespace/punctuation so it remains a valid dot-path
# segment. The parent's `numChildren` is incremented and a new edge is
# appended. Duplicate (parent_path, label) combinations are skipped.
inject_extra_leaves_into_graph <- function(graph, extras) {
  if (is.null(extras) || nrow(extras) == 0) return(graph)

  nodes <- graph %N>% as_tibble()
  edges <- graph %E>% as_tibble()

  for (i in seq_len(nrow(extras))) {
    parent_path <- as.character(extras$parent_path[i])
    new_label   <- as.character(extras$label[i])
    safe_token  <- gsub("[^A-Za-z0-9]+", "", new_label)
    new_name    <- paste0(parent_path, ".", safe_token)

    if (new_name %in% nodes$name) next

    parent_idx <- which(nodes$name == parent_path)
    if (length(parent_idx) == 0) {
      warning("inject_extra_leaves_into_graph: parent path not found: ",
              parent_path)
      next
    }
    parent_idx <- parent_idx[1]

    new_row <- nodes[parent_idx, , drop = FALSE]
    new_row$name <- new_name
    new_row$numChildren <- 0L
    if ("hgca_celltype_v1_majority" %in% colnames(new_row))
      new_row$hgca_celltype_v1_majority <- new_label
    if ("hgca_celltype_v1" %in% colnames(new_row))
      new_row$hgca_celltype_v1 <- new_label
    if ("label" %in% colnames(new_row))
      new_row$label <- new_label

    nodes <- dplyr::bind_rows(nodes, new_row)
    new_node_idx <- nrow(nodes)

    nodes$numChildren[parent_idx] <- nodes$numChildren[parent_idx] + 1L

    if (nrow(edges) > 0) {
      edge_row <- edges[1, , drop = FALSE]
      edge_row$from <- parent_idx
      edge_row$to   <- new_node_idx
      edges <- dplyr::bind_rows(edges, edge_row)
    } else {
      edges <- tibble::tibble(from = parent_idx, to = new_node_idx)
    }
  }

  tbl_graph(nodes = nodes, edges = edges, directed = TRUE)
}

# Helper: prune leaves listed in `drop_leaf_names` and recursively prune any
# now-empty internal ancestors. Operates on the existing graph and recomputes
# numChildren via ensure_required_node_fields() before returning.
prune_leaves_by_name <- function(graph, drop_leaf_names) {
  if (length(drop_leaf_names) == 0) return(graph)

  repeat {
    nodes <- graph %N>% as_tibble()
    drop_idx <- which(nodes$name %in% drop_leaf_names & nodes$numChildren == 0)
    if (length(drop_idx) == 0) break

    graph <- graph %N>% filter(!(row_number() %in% drop_idx))
    graph <- ensure_required_node_fields(graph)

    # After dropping the explicit list, also drop any internal nodes that no
    # longer have any descendants (numChildren == 0 but they weren't leaves
    # to begin with). We approximate this by re-running on names of nodes
    # that became childless but aren't in the original-leaf-name set; the
    # explicit drop_leaf_names is satisfied so we only need cascade-pruning
    # here.
    drop_leaf_names <- character(0)
    nodes_after <- graph %N>% as_tibble()
    # Cascade: any node whose name was originally an internal (had children)
    # but now has numChildren == 0 and isn't the root.
    became_orphan <- nodes_after$name[
      nodes_after$numChildren == 0 & nodes_after$name != "root"
    ]
    # Don't drop original leaves (we keep all leaves except the explicit list)
    # — but we don't have an "original leaf set" handy here; instead we trust
    # the caller's drop_leaf_names. To safely cascade-prune former internals
    # that are now empty, we only target nodes whose `name` contains a dot
    # (i.e. not the lineage root) and whose `hgca_celltype_v1_majority` is NA
    # (so we don't accidentally drop a real labeled leaf).
    if ("hgca_celltype_v1_majority" %in% colnames(nodes_after)) {
      cascade <- nodes_after$name[
        nodes_after$numChildren == 0 &
        is.na(nodes_after$hgca_celltype_v1_majority) &
        nodes_after$name != "root"
      ]
      if (length(cascade) == 0) break
      drop_leaf_names <- cascade
    } else {
      break
    }
  }

  graph
}

# Function to create a single ARBOL overlay plot showing
#   - Original (e.g. Taurus or Organoid) cell types
#   - NEW cell types added by HGCA label transfer
#   - Optional fourth category for query-only labels not in HGCA reference
#     (e.g. organoid D / M-X cells)
#
# Uses the Wong colorblind-safe palette per the Nature research figure guide
# (https://research-figure-guide.nature.com/figures/). Defaults match the
# Taurus vs HGCA convention used elsewhere in this project (vermillion for
# the original dataset, HCA blue for HGCA additions, light grey for absent).
#
# Categories per leaf (up to 4):
#   - "In <original_label> original" -> `original_color`
#   - "New from <reference_label>"   -> `reference_color`
#   - `extra_only_label`             -> `extra_only_color`
#   - "Absent in both"               -> `absent_color`
#
# Internal-node edges inherit the highest-priority category among descendant
# leaves with priority: original > reference > extra-only > absent. This
# keeps original-reached branches in the original color and only purely-new
# reference branches in the reference color.
#
# New (backward-compatible) parameters:
#   original_label / reference_label : strings used in legend and in the per-
#       leaf inline count labels (e.g. "Organoid 1,330  · HGCA 5,103").
#   original_color / reference_color / absent_color : hex codes for the four
#       categories (defaults preserve the Taurus / HGCA palette).
#   extra_leaves        : tibble(parent_path, label) of organoid-only or
#       HGCA-only leaves to inject into the dendrogram before plotting.
#   extra_only_labels   : character vector of leaf labels (matched after
#       normalize_label_for_taxonomy / lowercase / squish) that should be
#       categorized as `extra_only_label` regardless of their n_orig / n_trans.
#   extra_only_label    : legend label for that category.
#   extra_only_color    : hex code for that category.
#   drop_absent_in_both : when TRUE, leaves with no original and no transferred
#       cells are removed from the dendrogram entirely (along with any
#       internal nodes that become empty as a result) rather than rendered
#       in grey.
create_arbol_overlay <- function(graph,
                                 original_labels_vec,
                                 transferred_labels_vec,
                                 lineage_name,
                                 subset_clade = NULL,
                                 title = NULL,
                                 show_internal_labels = TRUE,
                                 show_cell_counts = TRUE,
                                 original_label = "Taurus",
                                 reference_label = "HGCA",
                                 original_color = "#D55E00",
                                 reference_color = "#0072B2",
                                 absent_color = "#E0E0E0",
                                 extra_leaves = NULL,
                                 extra_only_labels = NULL,
                                 extra_only_label = "Query only (not in HGCA)",
                                 extra_only_color = "#999999",
                                 drop_absent_in_both = FALSE,
                                 internal_label_paths = NULL) {
  cat("\n", strrep("=", 70), "\n")
  cat(sprintf("ARBOL OVERLAY: %s vs %s TRANSFER - %s\n",
              toupper(original_label), toupper(reference_label), lineage_name))
  cat(strrep("=", 70), "\n")

  # Category labels (parameterized so the same function powers Taurus, organoid, ...)
  cat_orig   <- sprintf("In %s original", original_label)
  cat_ref    <- sprintf("New from %s", reference_label)
  cat_absent <- "Absent in both"
  cat_extra  <- extra_only_label

  if (!is.null(subset_clade)) {
    cat("📐 Subsetting to clade:", subset_clade, "\n")
    graph <- subset_to_clade(graph, subset_clade)
    graph <- ensure_required_node_fields(graph)
  }

  if (!is.null(extra_leaves) && nrow(extra_leaves) > 0) {
    cat(sprintf("➕ Injecting %d extra leaves into graph\n", nrow(extra_leaves)))
    graph <- inject_extra_leaves_into_graph(graph, extra_leaves)
    graph <- ensure_required_node_fields(graph)
  }

  # Normalize helper (mirror of usage_comparison normalizer)
  norm <- function(x) {
    x <- as.character(x)
    x <- normalize_label_for_taxonomy(x)
    x <- gsub("\\n", " ", x)
    x <- gsub("\\r", " ", x)
    x <- stringr::str_squish(x)
    tolower(x)
  }

  # Build leaves_df from the current graph; factored so we can rebuild
  # after pruning absent-in-both leaves.
  extra_only_norm <- if (length(extra_only_labels) > 0) norm(extra_only_labels)
                     else character(0)

  od <- tibble::tibble(
    label = names(original_labels_vec),
    n_orig = as.integer(original_labels_vec)
  ) %>%
    dplyr::mutate(label_norm = norm(label)) %>%
    dplyr::group_by(label_norm) %>%
    dplyr::summarise(n_orig = sum(n_orig), .groups = "drop")

  td <- tibble::tibble(
    label = names(transferred_labels_vec),
    n_trans = as.integer(transferred_labels_vec)
  ) %>%
    dplyr::mutate(label_norm = norm(label)) %>%
    dplyr::group_by(label_norm) %>%
    dplyr::summarise(n_trans = sum(n_trans), .groups = "drop")

  build_leaves_df <- function(g) {
    nd <- g %N>% as_tibble() %>% dplyr::mutate(.idx = dplyr::row_number())
    li <- which(nd$numChildren == 0)
    ln <- nd$hgca_celltype_v1_majority[li]
    lnn <- norm(ln)

    tibble::tibble(leaf_idx = li,
                   cell_type = ln,
                   cell_type_norm = lnn) %>%
      dplyr::left_join(od %>% dplyr::select(label_norm, n_orig),
                       by = c("cell_type_norm" = "label_norm")) %>%
      dplyr::left_join(td %>% dplyr::select(label_norm, n_trans),
                       by = c("cell_type_norm" = "label_norm")) %>%
      dplyr::mutate(
        n_orig  = dplyr::if_else(is.na(n_orig),  0L, n_orig),
        n_trans = dplyr::if_else(is.na(n_trans), 0L, n_trans),
        category = dplyr::case_when(
          cell_type_norm %in% extra_only_norm  ~ cat_extra,
          n_orig  > 0                          ~ cat_orig,
          n_orig == 0 & n_trans > 0            ~ cat_ref,
          TRUE                                 ~ cat_absent
        )
      )
  }

  leaves_df <- build_leaves_df(graph)

  cat(sprintf(
    "   Leaves: %d  |  %s: %d   %s: %d   %s: %d   %s: %d\n",
    nrow(leaves_df),
    cat_orig,   sum(leaves_df$category == cat_orig),
    cat_ref,    sum(leaves_df$category == cat_ref),
    cat_extra,  sum(leaves_df$category == cat_extra),
    cat_absent, sum(leaves_df$category == cat_absent)
  ))

  # Optionally prune leaves with no signal in either original or transferred
  # (and cascade-prune any internal nodes that become empty).
  if (drop_absent_in_both) {
    nd_now <- graph %N>% as_tibble()
    drop_names <- nd_now$name[
      leaves_df$leaf_idx[leaves_df$category == cat_absent]
    ]
    n_before <- nrow(leaves_df)
    if (length(drop_names) > 0) {
      graph <- prune_leaves_by_name(graph, drop_names)
      leaves_df <- build_leaves_df(graph)
      cat(sprintf("🪴 Dropped %d absent-in-both leaves; %d remain\n",
                  n_before - nrow(leaves_df), nrow(leaves_df)))
    }
  }

  # Recompute node_data after any graph mutations (subset, inject, prune)
  node_data <- graph %N>% as_tibble() %>%
    dplyr::mutate(.tidygraph_node_index = dplyr::row_number())

  # Valid HGCA v1 terms can also be internal ontology nodes (for example,
  # CD4 Treg, cDC2, and Enterocytes). Preserve their direct counts so the
  # incoming branch reflects broad PanGI knowledge without falsely coloring
  # every more-specific child as present in PanGI.
  internal_signals <- tibble::tibble(
    node_idx = integer(),
    node_label = character(),
    n_orig = integer(),
    n_trans = integer(),
    category = character()
  )
  if (length(internal_label_paths) > 0) {
    internal_map <- tibble::tibble(
      node_label = names(internal_label_paths),
      name = as.character(internal_label_paths),
      label_norm = norm(names(internal_label_paths))
    ) %>%
      dplyr::distinct(name, .keep_all = TRUE)

    internal_signals <- node_data %>%
      dplyr::filter(numChildren > 0) %>%
      dplyr::select(node_idx = .tidygraph_node_index, name) %>%
      dplyr::inner_join(internal_map, by = "name") %>%
      dplyr::left_join(od, by = "label_norm") %>%
      dplyr::left_join(td, by = "label_norm") %>%
      dplyr::mutate(
        n_orig = dplyr::coalesce(n_orig, 0L),
        n_trans = dplyr::coalesce(n_trans, 0L),
        category = dplyr::case_when(
          n_orig > 0 ~ cat_orig,
          n_trans > 0 ~ cat_ref,
          TRUE ~ cat_absent
        )
      ) %>%
      dplyr::select(node_idx, node_label, n_orig, n_trans, category)
  }

  signal_df <- dplyr::bind_rows(
    leaves_df %>%
      dplyr::transmute(
        node_idx = leaf_idx,
        node_label = cell_type,
        n_orig,
        n_trans,
        category
      ),
    internal_signals
  )

  # Propagate to internal nodes by max-priority of any descendant signal:
  # original > reference > extra-only > absent
  # (Original-reached branches stay in the original color; only purely-new
  # reference branches go to the reference color; extra-only branches only
  # surface when no original/reference signal exists above them.)
  priority <- c(
    "absent" = 0L, "extra" = 1L, "reference" = 2L, "original" = 3L
  )
  cat_to_priority <- c()
  cat_to_priority[cat_absent] <- "absent"
  cat_to_priority[cat_extra]  <- "extra"
  cat_to_priority[cat_ref]    <- "reference"
  cat_to_priority[cat_orig]   <- "original"

  signal_cat_by_idx <- setNames(signal_df$category, signal_df$node_idx)

  node_categories <- vapply(seq_len(nrow(node_data)), function(i) {
    nm <- node_data$name[i]
    descendants <- which(startsWith(node_data$name, nm))
    desc_signals <- intersect(descendants, signal_df$node_idx)
    if (length(desc_signals) == 0) return(cat_absent)
    cats <- signal_cat_by_idx[as.character(desc_signals)]
    cats[is.na(cats)] <- cat_absent
    prio_keys <- cat_to_priority[cats]
    prio_vals <- priority[prio_keys]
    cats[which.max(prio_vals)]
  }, character(1))

  # Direct cell counts for both leaf and valid internal cell-type nodes.
  n_orig_by_idx  <- setNames(signal_df$n_orig,  signal_df$node_idx)
  n_trans_by_idx <- setNames(signal_df$n_trans, signal_df$node_idx)
  n_orig_node  <- ifelse(node_data$.tidygraph_node_index %in% signal_df$node_idx,
                         n_orig_by_idx[as.character(node_data$.tidygraph_node_index)],
                         NA_integer_)
  n_trans_node <- ifelse(node_data$.tidygraph_node_index %in% signal_df$node_idx,
                         n_trans_by_idx[as.character(node_data$.tidygraph_node_index)],
                         NA_integer_)
  internal_label_by_idx <- setNames(
    internal_signals$node_label, internal_signals$node_idx
  )
  internal_label_node <- ifelse(
    node_data$.tidygraph_node_index %in% internal_signals$node_idx,
    internal_label_by_idx[as.character(node_data$.tidygraph_node_index)],
    NA_character_
  )

  graph2 <- graph %N>% dplyr::mutate(
    overlay_category = node_categories,
    n_orig_leaf      = as.integer(n_orig_node),
    n_trans_leaf     = as.integer(n_trans_node),
    internal_node_label = internal_label_node
  )

  # Edge color = overlay_category of the "to" (child) node
  node_with_idx <- graph2 %N>% as_tibble() %>%
    dplyr::mutate(.tidygraph_node_index = dplyr::row_number())
  edge_data <- graph2 %E>% as_tibble() %>%
    dplyr::left_join(node_with_idx %>%
                       dplyr::select(.tidygraph_node_index, overlay_category),
                     by = c("to" = ".tidygraph_node_index"))
  graph3 <- graph2 %E>% dplyr::mutate(edge_category = edge_data$overlay_category)

  # Up-to-4-category colorblind-safe palette per Nature research figure guide
  # (Wong palette). Defaults: HCA Blue = #0072B2; Vermillion = #D55E00.
  cat_colors <- c()
  cat_colors[cat_absent] <- absent_color
  cat_colors[cat_orig]   <- original_color
  cat_colors[cat_ref]    <- reference_color
  cat_colors[cat_extra]  <- extra_only_color

  # Legend order: original -> reference -> extra-only -> absent.
  # Drop categories with zero leaves from the legend (e.g. omit "Absent in both"
  # when every taxonomy leaf is covered by at least one annotation source).
  cat_levels <- intersect(
    c(cat_orig, cat_ref, cat_extra, cat_absent),
    unique(leaves_df$category)
  )

  # Compute total cells for title context, including broad labels attached to
  # internal taxonomy nodes.
  total_orig  <- sum(signal_df$n_orig,  na.rm = TRUE)
  total_trans <- sum(signal_df$n_trans, na.rm = TRUE)

  if (is.null(title)) {
    title <- sprintf("%s author cell types vs. %s label transfer",
                     original_label, reference_label)
  }

  # Embed cell counts directly in leaf labels for guaranteed readability
  # (no risk of label-text collisions with separate geom_node_text counts).
  # Use thin-space + middle-dot separators for a cleaner look.
  fmt_n <- function(x) scales::comma(dplyr::coalesce(x, 0L))
  graph3 <- graph3 %N>% dplyr::mutate(
    internal_label_full = dplyr::case_when(
      numChildren == 0 ~ NA_character_,
      !is.na(internal_node_label) &
        dplyr::coalesce(n_orig_leaf, 0L) > 0 &
        dplyr::coalesce(n_trans_leaf, 0L) > 0 ~ sprintf(
          "%s  \u00B7  %s %s  \u00B7  %s %s",
          internal_node_label,
          original_label, fmt_n(n_orig_leaf),
          reference_label, fmt_n(n_trans_leaf)
        ),
      !is.na(internal_node_label) &
        dplyr::coalesce(n_orig_leaf, 0L) > 0 ~ sprintf(
          "%s  \u00B7  %s %s",
          internal_node_label, original_label, fmt_n(n_orig_leaf)
        ),
      !is.na(internal_node_label) &
        dplyr::coalesce(n_trans_leaf, 0L) > 0 ~ sprintf(
          "%s  \u00B7  %s %s",
          internal_node_label, reference_label, fmt_n(n_trans_leaf)
        ),
      TRUE ~ un_camel_case(
        ifelse(grepl("\\|", name),
               sub('.*\\|', '', name),
               sub('.*\\.', '', name)))
    ),
    leaf_label_full = dplyr::case_when(
      numChildren > 0 ~ NA_character_,
      overlay_category == cat_extra ~ sprintf(
        "%s  \u00B7  %s %s",
        hgca_celltype_v1_majority,
        original_label, fmt_n(n_orig_leaf)
      ),
      overlay_category == cat_orig & dplyr::coalesce(n_trans_leaf, 0L) > 0 ~ sprintf(
        "%s  \u00B7  %s %s  \u00B7  %s %s",
        hgca_celltype_v1_majority,
        original_label,  fmt_n(n_orig_leaf),
        reference_label, fmt_n(n_trans_leaf)
      ),
      overlay_category == cat_orig ~ sprintf(
        "%s  \u00B7  %s %s",
        hgca_celltype_v1_majority,
        original_label, fmt_n(n_orig_leaf)
      ),
      overlay_category == cat_ref ~ sprintf(
        "%s  \u00B7  %s %s",
        hgca_celltype_v1_majority,
        reference_label, fmt_n(n_trans_leaf)
      ),
      TRUE ~ as.character(hgca_celltype_v1_majority)
    )
  )

  # Plot sizing: long leaf labels (with embedded counts) + legend on the right
  # require lots of horizontal room. Some lineages have very long leaf names
  # ("Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells) · HGCA 208")
  # AND long internal-node names ("Conventional Dendritic Cells Type2c DC2")
  # that push off the left side. Compute width adaptively from the longest
  # rendered leaf label so nothing clips into the legend on the right.
  n_nodes <- nrow(node_data)
  plot_size <- calculate_plot_size(n_nodes)
  longest_leaf_chars <- suppressWarnings(max(
    nchar(graph3 %N>% as_tibble() %>%
            dplyr::filter(numChildren == 0) %>%
            dplyr::pull(leaf_label_full)),
    na.rm = TRUE
  ))
  if (!is.finite(longest_leaf_chars)) longest_leaf_chars <- 30
  plot_size$width  <- max(plot_size$width * 2.60,
                          16 + longest_leaf_chars * 0.20)
  plot_size$height <- plot_size$height * 1.10

  cat(sprintf("   📐 Plot size: %.1f × %.1f inches  (longest leaf label: %d chars)\n",
              plot_size$width, plot_size$height, longest_leaf_chars))

  # Nature requires Arial or Helvetica for all figure text
  # (https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
  # Use Helvetica throughout (available on macOS / Linux via fontconfig).
  fig_family <- "Helvetica"

  # Dendrogram layout (leaves evenly spaced, all rendered at the deepest level).
  # The x-axis tick labels are hidden so the visual depth of each leaf is not
  # mis-read as its taxonomy resolution.
  p <- ggraph(graph3, layout = "dendrogram") +
    geom_edge_diagonal(aes(color = factor(edge_category, levels = cat_levels)),
                       linewidth = 1.8, alpha = 0.95) +
    scale_edge_color_manual(values = cat_colors,
                            breaks = cat_levels,
                            drop = TRUE,
                            name = NULL) +
    # Leaf labels remain black per Nature's no-colored-text requirement;
    # category is encoded by branches and the legend.
    geom_node_text(
      aes(filter = numChildren == 0,
          label = leaf_label_full,
          alpha = factor(overlay_category, levels = cat_levels)),
      nudge_y = 0.1, vjust = 0.5, hjust = 0, size = 9.5,
      color = "black", fontface = "bold", family = fig_family
    ) +
    scale_alpha_manual(values = setNames(
      c(0.30, 1.00, 1.00, 1.00),
      c(cat_absent, cat_orig, cat_ref, cat_extra)
    ), guide = "none")

  if (show_internal_labels) {
    # Place internal labels OFFSET to the left (toward root) of their node so
    # they don't collide with the leaf labels at the right edge of the plot.
    # Use un_camel_case() to recover the original cell-type names that the
    # taxonomy stores as concatenated dot-path tokens.
    p <- p + geom_node_text(
      aes(filter = numChildren > 0,
          label = internal_label_full),
      nudge_y = -0.20, vjust = 0.5, hjust = 1,
      color = "black", size = 8.0, fontface = "italic",
      family = fig_family
    )
  }

  p <- p +
    theme_linedraw(base_size = 20, base_family = fig_family) +
    ggtitle(title) +
    scale_y_reverse() +
    guides(
      edge_color = guide_legend(
        title = NULL, override.aes = list(linewidth = 7),
        order = 1
      )
    ) +
    theme(
      # Hide both axes entirely - dendrogram visual depth would be misread as
      # taxonomy resolution otherwise (every leaf is forced to deepest level).
      text         = element_text(family = fig_family, color = "black"),
      axis.text.x  = element_blank(),
      axis.ticks.x = element_blank(),
      axis.title.x = element_blank(),
      axis.text.y  = element_blank(),
      axis.ticks.y = element_blank(),
      axis.title.y = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      plot.title = element_text(family = fig_family, hjust = 0.5,
                                face = "bold", size = 24, color = "black",
                                margin = ggplot2::margin(b = 16)),
      legend.position = "right",
      legend.text = element_text(family = fig_family, size = 26,
                                 color = "black", face = "bold"),
      legend.key.height = grid::unit(2.6, "lines"),
      legend.key.width  = grid::unit(3.0, "lines"),
      plot.margin = ggplot2::margin(t = 10, r = 18, b = 10, l = 18)
    ) +
    coord_flip(clip = "off") +
    # Push the right edge well past the deepest level so long leaf labels
    # (with embedded count text) don't clip into the legend area, and push
    # the left edge so long internal labels (e.g. "Conventional Dendritic
    # Cells Type2c DC2") don't clip into the plot frame.
    expand_limits(y = c(-13, 9))

  list(plot = p,
       width = plot_size$width,
       height = plot_size$height,
       leaves_df = leaves_df,
       category_counts = table(leaves_df$category))
}

# Test if run directly
if (sys.nframe() == 0) {
  cat("ARBOL Visualization Module\n\n")
  cat("Key Functions:\n")
  cat("  - subset_to_clade(): Subset taxonomy tree\n")
  cat("  - calculate_perclass_accuracy(): Load predictions and calculate metrics\n")
  cat("  - create_arbol_sideways(): Generate ARBOL dendrogram\n")
  cat("  - create_hgca_celltype_level1_arbol_plots(): Complete pipeline\n")
  cat("  - create_usage_comparison_arbol(): Before/after label transfer plots\n")
  cat("  - create_arbol_overlay(): SINGLE-PLOT overlay of Taurus original vs HGCA-new\n")
  cat("\nExample usage:\n")
  cat('  source("src/visualization/arbol.R")\n')
  cat('  create_hgca_celltype_level1_arbol_plots("configs/myeloid.yaml", "results/myeloid")\n')
}

