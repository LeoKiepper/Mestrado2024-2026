import pandas as pd

def plan_lag_feature_spacing(df: pd.DataFrame, lag_window: float, target_lag_memory_bytes: int, report: bool = True, exclude: list = []):
    df_filtered = df.drop(columns=exclude, errors='ignore')
    col_mem = df_filtered.memory_usage(deep=True, index=True) / len(df_filtered)
    total_row_mem = col_mem.sum()
    total_df_mem = df_filtered.memory_usage(deep=True, index=True).sum()
    n_rows = df_filtered.shape[0]
    per_lag_mem = total_row_mem * n_rows
    n_lags = int((target_lag_memory_bytes / per_lag_mem))
    if n_lags < 1:
        if report:
            print(f"Target memory too small for even one lag feature. Each lag feature requires {per_lag_mem/1e6:.2f} MB.")
        return None
    lag_spacing = lag_window / n_lags

    if report:
        print(f"Shape: {df_filtered.shape}\n")
        print("Memory usage per row (bytes) by column:")
        print("Col # | Column Name          | Data type       | Memory usage (bytes)")
        print("-" * 65)
        for idx, col in enumerate(col_mem.index):
            if col == 'Index':
                dd = df_filtered.index
                ii = 'Index'
            else:
                dd = df_filtered[col]
                ii = f"{idx:5d}"
            print(f"{ii} | {col:20} | {str(dd.dtype):15} | {col_mem[col]:18}")
        print("-" * 65)
        print(f"Estimated memory usage per row: {total_row_mem:.2f} bytes")
        print(f"Total memory usage (original dataset): {total_df_mem/1e6:.2f} MB")
        print(f"Target memory for new dataset: {target_lag_memory_bytes/1e6:.2f} MB")
        print(f"Number of lag features (excluding zero-delay): {n_lags}")
        print(f"Average delay between consecutive lag features: {lag_spacing:.3f} seconds")
    return lag_spacing
if __name__ == "__main__":
	# Generate example dataset
	import numpy as np
	import pandas as pd

	n_rows = 10000
	time = np.linspace(0, 9999, n_rows)
	data = {
		'T_CPU': np.random.normal(loc=50, scale=5, size=n_rows).astype(np.float32),
		'i_M1': np.random.normal(loc=1000, scale=100, size=n_rows).astype(np.int32),
		'i_M2': np.random.normal(loc=1000, scale=100, size=n_rows).astype(np.int32),
		'CPU_0': np.random.uniform(0, 1, n_rows).astype(np.float32),
		'CPU_1': np.random.uniform(0, 1, n_rows).astype(np.float32),
	}
	df = pd.DataFrame(data, index=time)
	df.index.name = 'Time'

	plan_lag_feature_spacing(df, lag_window=500, target_lag_memory_bytes=256e6)