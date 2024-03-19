import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from eval import calculate_accuracy, calculate_f1, calculate_precision, calculate_recall
from tqdm import tqdm
import pickle
import json
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def training(num_epochs, model, train_dataloader, val_dataloader,lr,  model_name=None, loss_weights = None, optimize=0, wd=1e-5):

    
    if optimize ==0:
        optimizer = Adam(model.parameters(), lr = lr, weight_decay = wd)
    if optimize ==1:
        optimizer = SGD(model.parameters(), lr = lr, weight_decay = wd)
        
    train_epoch_loss = {}
    val_epoch_loss = {}
    train_accuracy_l = {}
    train_f1_l = {}
    val_accuracy_l = {}
    val_f1_l = {}

    for epoch in tqdm(range(num_epochs), desc ='Training procedure'):

        model = model.to(device)
        model.train()
        train_loss = 0.0

        train_preds = []
        train_labels = []


        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, weight = loss_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_dataloader)
        train_epoch_loss[epoch] = train_loss

        train_accuracy = calculate_accuracy(train_labels, train_preds)
        train_accuracy_l[epoch] = train_accuracy

        train_f1 = calculate_f1(train_labels, train_preds)
        train_f1_l[epoch] = train_f1

        #validation

        model.eval()
        val_loss =0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_dataloader)
        val_epoch_loss[epoch] = val_loss

        val_accuracy = calculate_accuracy(val_labels, val_preds)
        val_accuracy_l[epoch] = val_accuracy

        val_f1 = calculate_f1(val_labels, val_preds)
        val_f1_l[epoch] = val_f1

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, train_f1: {train_f1:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, val_f1: {val_f1:.4f}')

    torch.save(model.state_dict(), f'data/new_checkpoints/{model_name}_checkpoint_epoch_{epoch + 1}.pt')

    with open(f'data/new_metrics/{model_name}_train.json', 'w') as f:
        json.dump({'train_epoch_loss': train_epoch_loss, 'train_accuracy_l': train_accuracy_l, 'train_f1_l': train_f1_l, 'val_epoch_loss': val_epoch_loss, 'val_accuracy_l': val_accuracy_l, 'val_f1_l': val_f1_l}, f)
    

def test(model, test_dataloader, model_name=None):
    model = model.to(device)
    model.eval()

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Calculate test metrics
    test_accuracy = calculate_accuracy(test_labels, test_preds)
    test_f1 = calculate_f1(test_labels, test_preds)
    precision = calculate_precision(test_labels, test_preds)
    recall = calculate_recall(test_labels, test_preds)

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1: {test_f1:.4f}')
    print(f'Test Precision: {precision}')
    print(f'Test Recall: {recall}')



    with open(f'data/new_metrics/{model_name}_test.json', 'w') as f:
        json.dump({'test_accuracy': test_accuracy, 'test_f1': test_f1, 'test_precision': precision, 'test_recall': recall}, f)
        
    return test_f1