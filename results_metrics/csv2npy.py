import pandas as pd
import numpy as np
import sys


def main(resultsfile):    
    # Load the CSV file
    df = pd.read_csv(resultsfile)
    
    # Get the unique patient IDs, metrics, and classes
    patients = df["patient_id"].unique()
    metrics = df["metric_type"].unique()
    classes = df["class"].unique()

    for metric in metrics:
        # Filter the DataFrame for the current metric
        metric_df = df[df["metric_type"] == metric]

        # Initialize an empty array to store the results
        results = np.zeros((len(patients), len(classes), 1))

        # Loop through each patient
        for i, patient in enumerate(patients):
            # Filter the DataFrame for the current patient
            patient_df = metric_df[metric_df["patient_id"] == patient]

            # Loop through each class
            for j, cls in enumerate(classes):
                # Filter the DataFrame for the current class
                class_df = patient_df[patient_df["class"] == cls]

                # Calculate the average result for the patient and class
                avg_result = np.mean(class_df["metric_value"])

                # Store the result in the results array
                results[i, j, 0] = avg_result

        # Save the results to a .npy file
        np.save(f"{metric}.npy", results)

if __name__ == "__main__":
    resultsfile = sys.argv[1]
    main(resultsfile)
