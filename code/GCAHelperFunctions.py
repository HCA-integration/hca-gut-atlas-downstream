def profile_function(adata, func, N_values, suppress_output=False, *args, **kwargs):
    import time, tracemalloc, matplotlib.pyplot as plt, numpy as np
    import contextlib, io
    N_sorted = sorted(N_values, reverse=True)
    times = {}
    mems = {}
    largest = N_sorted[0]
    idx = np.random.choice(adata.shape[0], largest, replace=False)
    current_subset = adata[idx,:]
    for N in N_sorted:
        if current_subset.shape[0] > N:
            current_subset = current_subset[np.random.choice(current_subset.shape[0], N, replace=False),:]
        tracemalloc.start()
        start = time.perf_counter()
        if suppress_output:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                func(current_subset, *args, **kwargs)
        else:
            func(current_subset, *args, **kwargs)
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times[N] = elapsed
        mems[N] = peak / 1024
    return times, mems

def plot_profile(time, mems): 
    sorted_N = sorted(times.keys())
    time_vals = [times[n] for n in sorted_N]
    mem_vals = [mems[n] for n in sorted_N]
    fig1 = plt.figure()
    plt.plot(sorted_N, time_vals)
    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.title('Time vs N')
    fig2 = plt.figure()
    plt.plot(sorted_N, mem_vals)
    plt.xlabel('N')
    plt.ylabel('Peak Memory (KB)')
    plt.title('Memory vs N')
    plt.show()