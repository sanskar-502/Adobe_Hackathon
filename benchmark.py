import time
import os
import psutil
from pathlib import Path
from process_pdfs import process_pdf_ml_enhanced

INPUT_DIR = Path("sample_dataset/pdfs")
OUTPUT_DIR = Path("sample_dataset/outputs")
RESULTS = []

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # in MB

def benchmark_pdf(pdf_file):
    print(f"Benchmarking: {pdf_file.name}")
    start_time = time.perf_counter()
    mem_before = get_memory_usage_mb()

    output_file = OUTPUT_DIR / f"{pdf_file.stem}.json"
    process_pdf_ml_enhanced(pdf_file, output_file)

    elapsed = time.perf_counter() - start_time
    mem_after = get_memory_usage_mb()

    memory_used = mem_after - mem_before
    RESULTS.append((pdf_file.name, elapsed, memory_used))

    print(f"Time: {elapsed:.2f}s | Memory: {memory_used:.2f}MB\n")

def run_benchmark():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("*.pdf"))

    for pdf in files:
        benchmark_pdf(pdf)

    print("\nBenchmark Results:")
    print("-" * 40)
    for name, t, m in RESULTS:
        print(f"{name:<30} Time: {t:.2f}s | Memory: {m:.2f}MB")
    print("-" * 40)

if __name__ == "__main__":
    run_benchmark()
