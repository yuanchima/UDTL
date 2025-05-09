import os
import re
import pandas as pd

# Define the directory containing the log files
log_dir = './checkpoint/canshutiaozheng5/'
save_dir = '.'

# Function to process a single log file
def process_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract parameters from first 34 lines
    params = {}
    for line in lines[:36]:
        if re.match(r'^\d{2}-\d{2} \d{2}:\d{2}:\d{2} ([^:]+): (.+)$', line):
            param_part = line[15:].strip()
            key, value = param_part.split(': ', 1)
            params[key.strip()] = value.strip()

    # Extract last epoch metrics
    last_metrics = {}
    for line in lines[-4:]:  # Check last 4 lines
        if 'source_train-Loss:' in line:
            parts = line.split('source_train-Loss:')[1].split('source_train-Acc:')
            last_metrics['source_train_loss'] = float(parts[0].strip())
            acc_part = parts[1].split(',')[0]
            last_metrics['source_train_acc'] = float(acc_part.strip())
        elif 'source_val-Loss:' in line:
            parts = line.split('source_val-Loss:')[1].split('source_val-Acc:')
            last_metrics['source_val_loss'] = float(parts[0].strip())
            acc_part = parts[1].split(',')[0]
            last_metrics['source_val_acc'] = float(acc_part.strip())
        elif 'target_val-Loss:' in line:
            parts = line.split('target_val-Loss:')[1].split('target_val-Acc:')
            last_metrics['target_val_loss'] = float(parts[0].strip())
            acc_part = parts[1].split(',')[0]
            last_metrics['target_val_acc'] = float(acc_part.strip())

    # Find best acc from save model lines
    best_acc = 0.0
    best_epoch = None
    for line in lines:
        if 'save best model epoch' in line:
            match = re.search(r'save best model epoch (\d+), acc ([\d.]+)', line)
            if match:
                acc = float(match.group(2))
                epoch = int(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch

    # Merge all results 
    result = {
        'file': file_path,
        'params': params, 
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        **last_metrics
    }
    
    return result

# Process all log files in the directory
def process_all_logs():
    results = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file == 'train.log':
                file_path = os.path.join(root, file)
                result = process_log_file(file_path)
                results.append(result)
    
    # Convert to DataFrame and handle params column
    df = pd.DataFrame(results)
    
    # Expand params dictionary into separate columns
    param_df = pd.DataFrame([d for d in df['params']])
    df = pd.concat([df.drop('params', axis=1), param_df], axis=1)
    
    return df

if __name__ == '__main__':
    df_results = process_all_logs()
    
    # Save to CSV
    output_path = os.path.join(save_dir, 'log_results.csv')
    df_results.to_csv(output_path, index=False)
    
    # Display results
    print("Results saved to:", output_path)
    print("\nDataFrame Preview:")
    print(df_results)