import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pdb

class BaseEvaluator:
    def __init__(self, args):
        self.gt_path = args.gt_path
        self.prediction_path = args.prediction_path
        self.llr_threshold = args.llr_threshold
        self.mode = args.mode
    
    def evaluate(self):
        print(f"Evaluating on {self.mode} mode.....")

        assert self.mode != None, "mode should be either batches (evaluate on all batches) or file (evaluate on a single file)"
        
        if self.mode in "batches":
            gt_labels, predicted_labels, scores = self.evaluate_all_batches()
        elif self.mode in "file":
            gt_labels, predicted_labels, scores = self.evaluate_single_file()
        
        self.print_metrics(gt_labels, predicted_labels, self.llr_threshold)
        
        # Finding the optimal llr scores for highest F1-score
        self.draw_llr_scores(scores)
        best_llr_threshold, optimal_predicted_labels = self.find_optimal_llr_threshold(gt_labels, scores)
        self.draw_confusion_matrix(gt_labels, optimal_predicted_labels)
        self.print_metrics(gt_labels, optimal_predicted_labels, best_llr_threshold)

    def evaluate_all_batches(self):
        gt_labels = []
        predicted_labels = []
        scores = []

        for batch_idx in range(1, len([f for f in os.listdir(self.prediction_path) if f != "figures"])+1):
            # Define the directory containing the JSON files
            prediction_dir = os.path.join(self.prediction_path, f'batch_{batch_idx}')
            
            print(f"Evaluating batch {batch_idx}...")
            # Read the batch csv file containing ground truth labels
            batch_csv_file = os.path.join(self.gt_path, f"batch_{batch_idx}.csv")
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
                        
                        if llr >= self.llr_threshold:
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
        
        return gt_labels, predicted_labels, scores
    
    def evaluate_single_file(self):
        gt_labels = []
        predicted_labels = []
        scores = []
        
        prediction_dir = self.prediction_path
        
        print(f"Evaluating {self.prediction_path}...")
        # Read the batch csv file containing ground truth labels
        batch_csv_file = self.gt_path
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
                    
                    if llr >= self.llr_threshold:
                        predicted_labels.append(1)
                    else:
                        predicted_labels.append(0)
                except:
                    llr = 0 # no CP predicted
                    predicted_labels.append(0)

                scores.append(llr)
                # Get the label from the ground truth csv file and add it to the gt_labels list
                segment_id = os.path.splitext(filename)[0] # Get the segment ID from the filename
                # pdb.set_trace()
                try:
                    label = batch_df.loc[batch_df['segment_id'] == segment_id]['label'].item() # Get the label from the csv
                except:
                    label = 0
                gt_labels.append(label)
    
        return gt_labels, predicted_labels, scores
    
    def print_metrics(self, gt_labels, predicted_labels, llr_threshold):
        print("----------------------------------------------------")
        print("LLR threshold: ", llr_threshold)
        # Calculate precision, recall, accuracy, and F1 score
        precision = precision_score(gt_labels, predicted_labels)
        recall = recall_score(gt_labels, predicted_labels)
        accuracy = accuracy_score(gt_labels, predicted_labels)
        f1 = f1_score(gt_labels, predicted_labels)        
        cr = classification_report(gt_labels, predicted_labels, target_names=['Non-CP', 'CP'], zero_division=0)
        
        # Print the evaluation metrics
        print('Precision: {:.2f}'.format(precision))
        print('Recall: {:.2f}'.format(recall))
        print('Accuracy: {:.2f}'.format(accuracy))
        print('F1 score: {:.2f}'.format(f1))
        print(f'Classification report: \n {cr}')
        print("---------------------------------------------------- \n")
    
    def draw_confusion_matrix(self, gt_labels, predicted_labels):
        figures_path = os.path.join(self.prediction_path, 'figures')
        Path(figures_path).mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(gt_labels, predicted_labels)
        # Display and save confusion matrix as image
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-CP', 'CP'])
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')        
        save_path = os.path.join(figures_path, 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saving confusion matrix in {save_path} \n")
        
    def draw_llr_scores(self, scores):
        figures_path = os.path.join(self.prediction_path, 'figures')
        Path(figures_path).mkdir(parents=True, exist_ok=True)
        x_values = list(range(1, len(scores) + 1))  # create x values from 1 to len(llr)
        # Create the plot
        plt.plot(x_values, scores)

        # Set the plot title and axis labels
        plt.title("Tracking LLR scores on the segments")
        plt.xlabel("Segments")
        plt.ylabel("Values")

        # Save the plot to an image file
        save_path = os.path.join(figures_path, 'llr_scores.png')
        plt.savefig(save_path)
        print(f"Saving LLR scores in {save_path} \n")
    
    def find_optimal_llr_threshold(self, gt_labels, scores): # find best threshold regarding f1
        # Create empty lists for storing evaluation metrics
        macro_avg_precision_list = []
        print("Finding the optimal llr threshold.....")
        predicted_labels_list = []
        # Loop through different threshold values and calculate evaluation metrics
        for threshold in tqdm(scores):
            # Convert the scores to binary predictions based on the threshold
            predicted_labels = [1 if llr >= threshold else 0 for llr in scores]

            # Calculate precision, recall, and F1 score
            macro_avg_precision = precision_score(gt_labels, predicted_labels, average='macro')
            
            # Add the evaluation metrics to their respective lists
            macro_avg_precision_list.append(macro_avg_precision)
            predicted_labels_list.append(predicted_labels)
        

        # Find the threshold that maximizes the F1 score
        best_threshold_index = np.argmax(macro_avg_precision_list)
        best_threshold = scores[best_threshold_index]
        print(f"---> Best LLR threshold: {best_threshold:.4f} (chosen from llr_scores.png)")       
        
        return best_threshold, predicted_labels_list[best_threshold_index]
