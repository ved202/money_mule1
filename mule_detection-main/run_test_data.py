import os
import sys

# Tell the system to save all outputs and data in specific folders
os.environ["MULE_OUTPUT_DIR"] = "testing_output"
os.environ["MULE_DATA_DIR"] = "testing_data"
os.environ["PYTHONUTF8"] = "1"

# Import the main pipeline
import main_pipeline

if __name__ == "__main__":
    print(f"\n============================================================")
    print(f"  RUNNING ADVANCED TEST DATA (saving to 'testing_output/' folder)")
    print(f"============================================================\n")

    # If no arguments are provided, automatically run the new advanced generator with GNN
    if len(sys.argv) == 1:
        print("No arguments provided. Automatically running the Advanced Synthetic Generator with GNN...")
        sys.argv.extend(["--source", "advanced_synthetic", "--with-gnn"])
        
    # Hand over execution to the main pipeline
    main_pipeline.run_pipeline()
