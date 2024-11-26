import sqlite3
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

# -----------------------------
# Connect to SQLite Database
# -----------------------------
db_path = "buy_habits.db"  # Path to your SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# -----------------------------
# Fetch Data from Database
# -----------------------------
def fetch_data(query):
    """Fetch data from the SQLite database using the provided query."""
    print("Fetching data...")
    return pd.read_sql_query(query, conn)

# Query the required columns for sorting
query = """
SELECT id, game_name, price, playtime, number_of_reviews
FROM buying_habits
WHERE price IS NOT NULL AND playtime IS NOT NULL
"""
df = fetch_data(query)

# -----------------------------
# Sampling and Chunking Methods
# -----------------------------
def get_full_dataset(df):
    """Return the full dataset."""
    return df

def get_sample(df, fraction):
    """Return a random sample of the dataset."""
    print(f"Sampling {fraction * 100}% of the dataset...")
    return df.sample(frac=fraction, random_state=42)

def get_chunks(df, num_chunks=10):
    """Split the full dataset into equal-sized chunks based on the number of chunks."""
    chunk_size = len(df) // num_chunks
    chunks = [df.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    if len(df) % num_chunks > 0:
        chunks[-1] = pd.concat([chunks[-1], df.iloc[-(len(df) % num_chunks):]])
    return chunks

# -----------------------------
# Sorting Algorithms
# -----------------------------
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def bubble_sort(arr):
    n = len(arr)
    for i in tqdm(range(n), desc="Bubble Sort Progress", leave=False):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def heap_sort(arr):
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

# -----------------------------
# Benchmarking Functions
# -----------------------------
def benchmark_sorting(df, column, sorting_fn, method_name, complexity):
    print(f"Benchmarking {method_name} on column '{column}'...")
    start_time = time.time()
    sorted_column = sorting_fn(df[column].tolist())
    end_time = time.time()
    return method_name, end_time - start_time, complexity

def benchmark_chunked_sorting(df, column, sorting_fn, method_name, complexity, num_chunks):
    chunks = get_chunks(df, num_chunks)
    total_time = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(benchmark_sorting, chunk, column, sorting_fn, f"{method_name} - Chunk {i+1}", complexity) 
                   for i, chunk in enumerate(chunks)]
        for future in concurrent.futures.as_completed(futures):
            _, elapsed_time, _ = future.result()
            total_time += elapsed_time
    return method_name, total_time, complexity

# -----------------------------
# Time Complexity Definitions
# -----------------------------
complexities = {
    "Bubble Sort": "O(n^2)",
    "Quicksort": "O(n log n)",
    "Merge Sort": "O(n log n)",
    "Heap Sort": "O(n log n)",
    "Built-in Sort": "O(n log n)"
}

# -----------------------------
# Benchmarking All Algorithms
# -----------------------------
results = []

datasets = [
    ("Full Dataset", get_full_dataset(df)),
    ("Sampled 25%", get_sample(df, 0.25)),
    ("Chunked", get_full_dataset(df))
]

algorithms = [
    ("Quicksort", quicksort),
    ("Bubble Sort", bubble_sort),
    ("Merge Sort", merge_sort),
    ("Heap Sort", heap_sort),
    ("Built-in Sort", sorted)
]

num_chunks = 10
for method, dataset in tqdm(datasets, desc="Processing Datasets"):
    for algorithm, fn in algorithms:
        column_to_sort = "price"
        complexity = complexities[algorithm]
        if method == "Chunked":
            method_name, elapsed_time, complexity = benchmark_chunked_sorting(dataset, column_to_sort, fn, f"{method} - {algorithm}", complexity, num_chunks)
        else:
            method_name, elapsed_time, complexity = benchmark_sorting(dataset, column_to_sort, fn, f"{method} - {algorithm}", complexity)
        results.append((method_name, elapsed_time, complexity))

# -----------------------------
# Visualize Results
# -----------------------------
# Convert results to a DataFrame
# Convert results to a DataFrame
# -----------------------------
# Save and Visualize Results
# -----------------------------
print("\nVisualizing results...")

# Convert results to a DataFrame for easier handling
results_df = pd.DataFrame(results, columns=["Method", "Time", "Complexity"])

# Define color coding for dataset types (not algorithms)
dataset_colors = {
    "Full Dataset": 'lightblue',
    "Sampled 25%": 'lightgreen',
    "Chunked": 'lightcoral'
}

# Assign colors based on dataset type (extracted from the "Method" column)
results_df['Color'] = results_df['Method'].apply(lambda x: dataset_colors.get(x.split(' - ')[0], 'gray'))

# Create a horizontal bar chart
plt.figure(figsize=(14, 8))
bars = plt.barh(results_df["Method"], results_df["Time"], color=results_df["Color"])
plt.xscale('log')  # Logarithmic scale for better readability of time differences
plt.xlabel("Time (seconds, log scale)", fontsize=12)
plt.ylabel("Sorting Method and Dataset Type", fontsize=12)
plt.title("String Comparison", fontsize=14)

# Annotate each bar dynamically
for index, row in results_df.iterrows():
    bar_width = row["Time"]
    text_x = bar_width * 1.1 if bar_width < 0.1 else bar_width * 0.9  # Position outside for small bars
    ha = 'left' if bar_width < 0.1 else 'right'  # Align text outside or inside based on bar width

    plt.text(
        text_x,  # Adjust X-position based on bar width
        index,   # Y-coordinate
        f"{row['Time']:.6f}s\n{row['Complexity']}",  # Annotation text
        va='center',  # Center vertically
        ha=ha,  # Dynamic horizontal alignment
        fontsize=10   # Font size for annotations
    )

# Adjust layout and save the plot
plt.tight_layout()
output_path = "string_comparison.png"
plt.savefig(output_path)
print(f"Plot saved as: {output_path}")
plt.show()
