import requests
from datetime import datetime
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import norm
import json
import matplotlib.pyplot as plt
import os
import time

paths = [
    # "../data/yearwise/spy_options_data_16.json",
    # "../data/yearwise/spy_options_data_17.json",
    # "../data/yearwise/spy_options_data_18.json",
    # "../data/yearwise/spy_options_data_19.json",
    "../data/yearwise/spy_options_data_20.json",
    "../data/yearwise/spy_options_data_21.json",
    "../data/yearwise/spy_options_data_22.json",
    "../data/yearwise/spy_options_data_23.json",
]

# Output file path
masterjson_path = "../data/ivs_20_23.json"

# Memory-efficient processing with buffered writes
def process_files(input_paths, output_path, buffer_size=5000):
    """
    Process JSON files efficiently with buffered writing to prevent memory issues.
    
    Args:
        input_paths: List of input JSON file paths
        output_path: Path to write the results
        buffer_size: Number of records to buffer before writing
    """
    # Clear output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    buffer = []
    total_count = 0
    start_time = time.time()
    
    for path in input_paths:
        print(f"Processing {path}...")
        file_count = 0
        
        # Load each file once
        with open(path) as f:
            data = json.load(f)
            
            # Process all objects
            for obj in data:
                for sub in obj:
                    buffer.append(sub)
                    file_count += 1
                    
                    # When buffer is full, write to file
                    if len(buffer) >= buffer_size:
                        with open(output_path, "a") as out_file:
                            for item in buffer:
                                json.dump(item, out_file)
                                out_file.write("\n")
                        # Clear buffer after writing
                        buffer = []
        
        total_count += file_count
        print(f"  Added {file_count} objects from {path}")
    
    # Write any remaining items in buffer
    if buffer:
        with open(output_path, "a") as out_file:
            for item in buffer:
                json.dump(item, out_file)
                out_file.write("\n")
    
    elapsed = time.time() - start_time
    print(f"\nComplete! Processed {total_count} total objects in {elapsed:.2f} seconds")
    print(f"Output written to: {output_path}")

# Run the processing function
if __name__ == "__main__":
    process_files(paths, masterjson_path, buffer_size=5000)