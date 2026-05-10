from datasets import two_moons
import torch.utils.data as data
import torch
import zuko

def dataset(flag="train", prediction_length=50, context_length=50, data: torch.Tensor = None, stride: int = 1, val_portion: float = 0.1) -> data.Dataset:
    #data is torch.tensor of shape (N, T) and is univariate time series data
    #createing training examples
    if flag == "train":
        split_index = int(data.shape[1] * (1 - val_portion))
        training_set =data[: , :split_index]
        
        contexts,ground_truths =[],[]
        for i in range(0, training_set.shape[1], stride):
            if i+context_length + prediction_length > training_set.shape[1]:
                break  
            context = training_set[:, i:i+context_length]
            ground_truth = training_set[:, i+context_length:i+context_length+prediction_length]
            
            contexts.append(context)
            ground_truths.append(ground_truth)
        
        context_tensor = torch.stack(contexts)
        ground_truth_tensor = torch.stack(ground_truths)
        
    if flag=="val":
        split_index = int(data.shape[1] * (1 - val_portion))
        validation_set =data[: , split_index:]
        
        contexts,ground_truths =[],[]
        for i in range(0, validation_set.shape[1], stride):
            if i+context_length + prediction_length > validation_set.shape[1]:
                break  
            context = validation_set[:, i:i+context_length]
            ground_truth = validation_set[:, i+context_length:i+context_length+prediction_length]
            
            contexts.append(context)
            ground_truths.append(ground_truth)
        
        context_tensor = torch.stack(contexts)
        ground_truth_tensor = torch.stack(ground_truths)
    
    
    return data.TensorDataset(context_tensor, ground_truth_tensor)
            
        
    
    

def train(prediction_length=50, context_length=50, batch_size=128, epochs=1000):
    flow = zuko.flows.NSF(features=prediction_length, transforms=3, context=context_length, hidden_features=(64, 64))
    