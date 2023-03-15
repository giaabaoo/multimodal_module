import json
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from tools.helper import *
import matplotlib.pyplot as plt
import numpy as np
import pdb

if __name__ == '__main__':
    ##### Defining arguments #####
    parser = argparse.ArgumentParser(
        "Evaluating UCPD results on multi-modal data", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Define the LLR threshold
    LLR_THRESHOLD = 1
    # Create empty lists for storing the actual and predicted labels
    gt_labels = []
    predicted_labels = []
    scores = []

    if args.batch_idx != -1:
        for batch_idx in range(1, len(os.listdir("output/batches"))+1):
            # Define the directory containing the JSON files
            prediction_dir = f'/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/output/batches/batch_{batch_idx}'
            print(f"Evaluating batch {batch_idx}...")
            # Read the batch csv file containing ground truth labels
            batch_csv_file = f"/home/dhgbao/Research_Monash/code/my_code/unsupervised_approach/multimodal_module/data/batches/batch_{batch_idx}.csv"
            batch_df = pd.read_csv(batch_csv_file)

            # Loop through the JSON files in the directory
            for filename in os.listdir(prediction_dir):
                if filename.endswith('.json'):
                    # Load the JSON file
                    with open(os.path.join(prediction_dir, filename), 'r') as f:
                        data = json.load(f)

                    # Extract the LLR value and convert it to a binary prediction
                    try:
                        llr = data['final_cp_llr'][0][0]
                        
                        # pdb.set_trace()
                        if llr >= LLR_THRESHOLD:
                            predicted_labels.append(1)
                        else:
                            predicted_labels.append(0)
                    except:
                        llr = 0 # no CP predicted
                        predicted_labels.append(0)

                    scores.append(llr)
                    # Get the label from the ground truth csv file and add it to the gt_labels list
                    segment_id = os.path.splitext(filename)[0] # Get the segment ID from the filename
                    label = batch_df.loc[batch_df['segment_id'] == segment_id]['label'].item() # Get the label from the csv
                    gt_labels.append(label)

    print("#########################################")
    print("LLR threshold: ", LLR_THRESHOLD)
    # Calculate precision, recall, accuracy, and F1 score
    precision = precision_score(gt_labels, predicted_labels)
    recall = recall_score(gt_labels, predicted_labels)
    accuracy = accuracy_score(gt_labels, predicted_labels)
    f1 = f1_score(gt_labels, predicted_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(gt_labels, predicted_labels)

    # Display and save confusion matrix as image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-CP', 'CP'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    x_values = list(range(1, len(scores) + 1))  # create x values from 1 to len(llr)
    # Create the plot
    plt.plot(x_values, scores)

    # Set the plot title and axis labels
    plt.title("Tracking LLR scores on the segments")
    plt.xlabel("Segments")
    plt.ylabel("Values")

    # Save the plot to an image file
    plt.savefig("llr_scores.png")

    # Print the evaluation metrics
    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('Accuracy: {:.2f}'.format(accuracy))
    print('F1 score: {:.2f}'.format(f1))
    
    # Create empty lists for storing evaluation metrics
    precision_list = []
    recall_list = []
    f1_list = []
    acc_list = []
    cr_list = []

    # Loop through different threshold values and calculate evaluation metrics
    for threshold in scores:
        # Convert the scores to binary predictions based on the threshold
        predicted_labels = [1 if llr >= threshold else 0 for llr in scores]

        # Calculate precision, recall, and F1 score
        precision = precision_score(gt_labels, predicted_labels)
        recall = recall_score(gt_labels, predicted_labels)
        f1 = f1_score(gt_labels, predicted_labels)
        acc = accuracy_score(gt_labels, predicted_labels)
        cr = classification_report(gt_labels, predicted_labels, target_names=['Non-CP', 'CP'], zero_division=0)

        # Add the evaluation metrics to their respective lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        acc_list.append(acc)
        cr_list.append(cr)

    # Find the threshold that maximizes the F1 score
    best_threshold_index = np.argmax(f1_list)
    best_threshold = scores[best_threshold_index]

    print("#########################################")
    print(f"Best LLR threshold: {best_threshold:.4f} (chosen from llr_scores.png)")
    print(f"Precision: {precision_list[best_threshold_index]:.2f}")
    print(f"Recall: {recall_list[best_threshold_index]:.2f}")
    print(f"F1 score: {f1_list[best_threshold_index]:.2f}")
    print(f"Accuracy: {acc_list[best_threshold_index]:.2f}")
    print(f"Classification report: {cr_list[best_threshold_index]}")
    print("Saving confusion matrix as confusion_matrix.png")
