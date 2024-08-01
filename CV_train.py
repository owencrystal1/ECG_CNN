import torch 
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from sklearn.preprocessing import MinMaxScaler, label_binarize
import neurokit2 as nk
import json
import os
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, classification_report, auc
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset
import ML_train

class SoftMaxCELoss(nn.Module):
        def __init__(self):
            super(SoftMaxCELoss, self).__init__()

        def forward(self, predictions, targets):
            """
            Compute the cross-entropy loss given softmax probabilities.
            
            Inputs:
                predictions: The output of a softmax layer (probabilities).
                targets: The ground truth labels.
            
            Output:
                torch.Tensor: The computed cross-entropy loss.
            """
            # Ensure targets are in long format
            if targets.dtype != torch.long:
                targets = targets.long()

            # Compute the negative log likelihood loss using the probabilities
            # Gather the probabilities for the true classes
            log_probabilities = torch.log(predictions + 1e-10)  # Adding epsilon for numerical stability
            target_log_probs = log_probabilities.gather(dim=1, index=targets.unsqueeze(1))
            
            # Calculate the loss
            loss = -target_log_probs.mean()  # Mean across the batch (calculated for each batch)

            return loss



def train_ECG_CNN(df_ecg, focused_leads, directory):

    with open('./hyperparameters.json') as in_file:
        
        hyperparameters = json.load(in_file)

    params = {
        'batch_size': hyperparameters['batch_size'],
        "weight_decay": hyperparameters['weight_decay'],
        'epochs': 1000,
        'learning_rate_base': hyperparameters['base_lr'],
        'learning_rate_max': hyperparameters['max_lr'],
        'step_up_size': hyperparameters['step_up_size'],
        'dropout_rate': hyperparameters['dropout_rate'],
        'kernel_size': hyperparameters['filter_size'],
        'num_conv_layers': hyperparameters['num_conv_layers'],
        'first_layer_filters': hyperparameters['first_layer_filters'],
    }


    logger = open(os.path.join('./grad_accum_8_random2.txt'), 'a')
    print('Training Parameters:', file=logger)
    print('-' * 10, file=logger)
    print(params, file=logger)
    print('-' * 10, file=logger)
    logger.flush()

    batch_size = params['batch_size']
    epochs = params['epochs']
    weight_decay = params['weight_decay']
    learning_rate_base = hyperparameters['base_lr']
    learning_rate_max = hyperparameters['max_lr']
    dropout_rate = params['dropout_rate']
    step_up_size = params['step_up_size']
    kernel_size = params['kernel_size']
    num_conv_layers = params['num_conv_layers']
    first_layer_filters = params['first_layer_filters']
    num_classes = 3


    df_ecg = df_ecg.fillna(0)
    X = df_ecg

    y = df_ecg['Diagnosis']

    def load_data(file_name):
        with open(directory + file_name + '.pickle', 'rb') as f:
            data = pickle.load(f)
            nums = data.values()
            all_leads = [data[key] for key in focused_leads]
        return torch.tensor(all_leads, dtype=torch.float32)

    samples = []
    labels = []
    pt_ids = []

    for index,row in X.iterrows():
        sample = load_data(row['File_Name'])
        if sample.shape[1] == 5000:

            sample1 = sample[:,:2500]
            samples.append(sample1)
            labels.append(row['Diagnosis'])
            pt_ids.append(row['Echo_File'])


            # sample2 = sample[:,2500:]
            # samples.append(sample2)
            # labels.append(row['Diagnosis'])
            # pt_ids.append(row['Echo_File'])
        else:
            samples.append(sample)
            labels.append(row['Diagnosis'])
            pt_ids.append(row['Echo_File'])

    samples = torch.stack(samples)
    

    class ECGDataset(Dataset):
        def __init__(self, samples, labels):
            self.samples = samples
            self.labels = labels

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            x = self.samples[idx] #2x2500
            scaled_tensor = torch.empty_like(x)

            # filter x using PT function
            for i in range(x.shape[0]):
                
                # filtering each individual channel/lead signal
                ch1 = nk.ecg_clean(x[i,:], sampling_rate=500, method="pantompkins1985") # pre processing ECG signal
                #ch1 = x[i,:]

                # save channel signal into new tensor to be normalized
                scaled_tensor[i,:] = torch.tensor(ch1)

            # min/max normalization 
            scaler = MinMaxScaler()

            for i in range(x.shape[0]):
                lead = scaled_tensor[i,:].reshape(-1,1)

                scaled_lead = scaler.fit_transform(lead)
                scaled_tensor[i,:] = torch.tensor(scaled_lead.flatten(), dtype=torch.float32)
                noise = np.random.normal(0, 0.1, scaled_tensor[i,:].shape)

                # Add noise to the tensor
                scaled_tensor[i,:] = scaled_tensor[i,:] + noise

            y = self.labels[idx]
            y = torch.tensor(y)


            return scaled_tensor, y

    class ECG_CNN(nn.Module):
        def __init__(self, num_conv_layers=num_conv_layers, in_channels=2, first_layer_filters=first_layer_filters, num_classes=num_classes, kernel_size=kernel_size):
            super(ECG_CNN, self).__init__()

            self.num_conv_layers = num_conv_layers
            self.in_channels = in_channels
            self.first_layer_filters = first_layer_filters

            # Initialize the convolutional layers dynamically
            conv_layers = []
            current_filters = first_layer_filters # 128
            for i in range(num_conv_layers): # 10

                if i % 2 != 0:
                    current_filters = int(current_filters)*2 # reset back to last number of filters for odd numbered layers (repeated num filters)
                    out_channels = int(current_filters)

                    # add max pooling layer at every other conv block
                    conv_layers.extend([
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                        nn.MaxPool1d(kernel_size=2,stride=2),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ])
                else: # add conv block without max pooling
                    out_channels = int(current_filters)
                    conv_layers.extend([
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ])

                in_channels = out_channels  # Update in_channels for the next layer
                current_filters /= 2  # Halve the number of filters for the next layer

            self.conv_blocks = nn.Sequential(*conv_layers)
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)

            self._initialize_fc_layers()

        def _initialize_fc_layers(self):
            # Compute output size dynamically
            example_input = torch.randn(16, 2, 2500)  # Example input tensor
            with torch.set_grad_enabled(True):
                conv_output = self.conv_blocks(example_input)
                max_pool_output = self.global_max_pool(conv_output)
                conv_output_shape = max_pool_output.shape[1] # Number of channels = 8



            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_shape, 16),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, num_classes)
            )
        
        def forward(self, x):
            # Convolutional layers
            x = self.conv_blocks(x)

            x = self.global_max_pool(x)

            #x, _ = torch.max(x, dim=1, keepdim=True)
            
            x = x.squeeze(-1) 
            #x = x.squeeze(1) 
            

            # Fully connected layers
            x = self.fc_layers(x)

            x = F.softmax(x, dim=-1)

            return x


    # Define the number of splits
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    samples = np.array(samples)  # Convert to numpy array
    labels = np.array(labels)
    

    # Perform cross-validation
    phase = 'train'

    for fold, (train_index, val_index) in enumerate(kf.split(samples)):

        print(f"Fold {fold+1}")


        logger = open(os.path.join('./CV_{}_log_v2.txt'.format(fold)), 'a')
        print('Training Parameters:', file=logger)
        print('-' * 10, file=logger)
        print(params, file=logger)
        print('-' * 10, file=logger)
        logger.flush()

        # training and validation subsets
        train_samples, train_labels = samples[train_index], labels[train_index]
        val_samples, val_labels = samples[val_index], labels[val_index]

        train_dataset = ECGDataset(torch.tensor(train_samples), train_labels)
        val_dataset = ECGDataset(torch.tensor(val_samples), val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if phase == 'train':
        
            # Initialize model, criterion, optimizer, etc.
            torch.cuda.manual_seed(42)
            model = ECG_CNN(num_conv_layers=num_conv_layers, first_layer_filters=first_layer_filters, kernel_size=kernel_size).to(device)
            model.to(device)
            criterion = SoftMaxCELoss()
            optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
            scheduler = CyclicLR(optimizer, base_lr=learning_rate_base, max_lr=learning_rate_max, step_size_up=step_up_size)
            accumulation_steps = 8

            # Zero the parameter gradients
            optimizer.zero_grad()
            num_epochs = 1000
            train_losses = []
            val_losses = []

            early_stopping_patience = 50
            early_stopping_counter = 0
            best_val_loss = float('inf')
            best_epoch = 0
            best_model_state = None

            for epoch in range(num_epochs):
                model.train()  # Set model to training model
                running_train_loss = 0.0

                for i, batch in enumerate(train_loader):
                    inputs, labels = batch

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(True):
                    
                        # Forward pass
                        outputs = model(inputs)
                    
                        # Compute loss
                        loss = criterion(outputs, labels)

                        # Normalize the loss to account for the accumulation steps
                    
                        # Backward pass and optimize
                        loss.backward()

                        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                            optimizer.step()
                            optimizer.zero_grad()
                        #optimizer.step()
                        running_train_loss += loss.item()
                    
                    # Track the training loss
                    #running_train_loss += loss.item()
                
                # Calculate average training loss for the epoch
                epoch_train_loss = running_train_loss / len(train_loader)
                train_losses.append(epoch_train_loss)
                
                # Validation phase
                model.eval()  # Set model to evaluation mode
                running_val_loss = 0.0
                correct = 0
                total = 0

                # have a softmax within the model ten create a custom loss functino that uses that softmax output, not further applying softmax twice 
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item() # counting te number of correct values
                        val_loss = criterion(outputs, labels)
                        running_val_loss += val_loss.item()
                
                # Calculate average validation loss and accuracy for the epoch
                epoch_val_loss = running_val_loss / len(val_loader)
                val_losses.append(epoch_val_loss)
                val_acc = correct / total
                
                # Update the learning rate scheduler
                scheduler.step(epoch_val_loss)
                
                # Early stopping logic - new best epoch
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_epoch = epoch
                    best_model_state = model.state_dict()
                    print('Saving best model at epoch:', best_epoch+1)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')

                        # Restore the best model state
                        model.load_state_dict(best_model_state)

                        true_labels = []
                        pred_labels = []
                        score_0 = []
                        score_1 = []
                        score_2 = []
                        pt_ids = []

                        # Evaluate on test set (optional)
                        model.eval()

                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                inputs = inputs.to(device)
                                labels = labels.to(device)
                                outputs = model(inputs)

                                score_0_batch = outputs[:, 0]
                                score_1_batch = outputs[:, 1]
                                score_2_batch = outputs[:, 2]
                                
                                true_labels.extend(labels.data.tolist())
                                pred_labels.extend(predicted.tolist())
                                score_0.extend(score_0_batch.tolist())
                                score_1.extend(score_1_batch.tolist())
                                score_2.extend(score_2_batch.tolist())


                        output_df = pd.DataFrame(columns=['label','AMY','HCM','HTN'])
                        output_df.label = true_labels
                        output_df.AMY = score_0
                        output_df.HCM = score_1
                        output_df.HTN = score_2
                        df = output_df
                        df.to_csv(f'model_fold_IDs_{fold+1}_v2.csv')

                        classes = ['AMY','HCM','HTN']
                        # Iterate over each class
                        for i in range(3):
                            # Compute ROC curve and ROC area for each class
                            fpr, tpr, _ = roc_curve(df['label'], df[f'{classes[i]}'], pos_label=i)
                            roc_auc = auc(fpr, tpr)
                            print(roc_auc, file=logger)
                        break


                # Print epoch statistics
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}')
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}', file=logger)
        
            # Plotting the training and validation losses
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Cross Entropy Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.savefig(f'model_fold_{fold+1}.png')

            # Save the model if necessary
            torch.save(model.state_dict(), f'model_fold_{fold+1}.pth')

        elif phase == 'test':
            model = ECG_CNN(num_conv_layers=num_conv_layers, first_layer_filters=first_layer_filters, kernel_size=kernel_size)
            model.to(device)

            model.load_state_dict(torch.load(f'ECG_CNN_results/model_fold_{fold+1}.pth'))

            true_labels = []
            pred_labels = []
            score_0 = []
            score_1 = []
            score_2 = []

            # Evaluate on test set (optional)
            model.eval()

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    score_0_batch = outputs[:, 0]
                    score_1_batch = outputs[:, 1]
                    score_2_batch = outputs[:, 2]
                    
                    true_labels.extend(labels.data.tolist())
                    pred_labels.extend(predicted.tolist())
                    score_0.extend(score_0_batch.tolist())
                    score_1.extend(score_1_batch.tolist())
                    score_2.extend(score_2_batch.tolist())


            output_df = pd.DataFrame(columns=['label','AMY','HCM','HTN'])
            output_df.label = true_labels
            output_df.AMY = score_0
            output_df.HCM = score_1
            output_df.HTN = score_2
            df = output_df
            df.to_csv(f'model_fold_IDs_{fold+1}.csv')

            #df = pd.read_csv('./model_all_folds.csv')

            y_true = df.label
            y_label = label_binarize(y_true.astype(int), classes=[0,1,2])
            

            columns_to_keep = ['AMY', 'HCM', 'HTN']

            probabilities = df[columns_to_keep].copy()

            # Find optimal probability threshold --> if its above this threshold, then we consider it positive 
            thresholds = ML_train.Find_Optimal_Cutoff(y_label, probabilities.values, 3)

            # make new predictions based on thresolds
            thresh_preds = ML_train.generate_metrics(y_true.astype(int), probabilities.values, thresholds)

            # get new performance metrics using new predictions
            ML_train.get_performance_metrics(y_true.astype(int), np.array(thresh_preds))

            # ROCs
            ML_train.get_metrics(y_true.astype(int), probabilities.values, 3, focused_leads, f'ECG_CNN_all_folds')

