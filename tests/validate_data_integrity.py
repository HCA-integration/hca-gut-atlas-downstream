# If you have required metadata variables for your downstream analysis of interest, you can 
# use the below function to validate that these metadata are filled for each dataset in the object

def validate_concat_object(dataset, required_columns=None, check_na_columns=None):
    if required_columns is None:
        required_columns = ["dataset_id", "donor_id", "sample_id", "author_cell_type"]
    if check_na_columns is None:
        check_na_columns = ["donor_id", "sample_id", "author_cell_type"]
    report = {"valid": True, "messages": []}
    # Check for missing required columns
    missing_columns = [col for col in required_columns if col not in dataset.obs.columns]
    if missing_columns:
        report["valid"] = False
        report["messages"].append(f"Missing required columns in obs: {missing_columns}")
    # Check for NA in key columns per dataset_id in obs
    for col in check_na_columns:
        if col in dataset.obs.columns:
            na_counts = dataset.obs.groupby('dataset_id')[col].apply(lambda x: x.isna().sum())
            if na_counts.any():
                for id, count in na_counts.iteritems():
                    if count > 0:
                        report["messages"].append(f"{count} NAs in column '{col}' for dataset_id '{id}'")
        else:
            report["messages"].append(f"Column '{col}' not found in obs.")
    # Calculate total counts per cell and check for zeros per dataset_id
    if 'total_counts' not in dataset.obs.columns:
        if scipy.sparse.issparse(dataset.X):
            dataset.obs['total_counts'] = dataset.X.sum(axis=1).A1  # Convert sparse matrix sum to a numpy array
        else:
            dataset.obs['total_counts'] = dataset.X.sum(axis=1)
    zero_counts = dataset.obs.groupby('dataset_id')['total_counts'].apply(lambda x: (x == 0).sum())
    for id, count in zero_counts.iteritems():
        if count > 0:
            report["messages"].append(f"{count} cells with zero total counts for dataset_id '{id}'")
    return report


