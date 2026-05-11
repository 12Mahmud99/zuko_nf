import torch
import numpy as np
import torch.utils.data as torch_data

def two_moons(n: int, sigma: float = 1e-1) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 2 * torch.pi * torch.rand(n)
    label = (theta > torch.pi).float()

    x = torch.stack(
        (
            torch.cos(theta) + label - 1 / 2,
            torch.sin(theta) + label / 2 - 1 / 4,
        ),
        axis=-1,
    )

    return torch.normal(x, sigma), label

def extraxct_data(path):
    data = np.load(path)
    return torch.from_numpy(data["positions"])


def dataset(flag="train", prediction_length=50, context_length=50, data: torch.Tensor = None, stride: int = 1, val_portion: float = 0.1) -> torch_data.Dataset:
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
            
            for s in range(context.shape[0]):
                contexts.append(context[s])    
                ground_truths.append(ground_truth[s]) 
        
    if flag=="val":
        split_index = int(data.shape[1] * (1 - val_portion))
        validation_set =data[: , split_index:]
        
        contexts,ground_truths =[],[]
        for i in range(0, validation_set.shape[1], stride):
            if i+context_length + prediction_length > validation_set.shape[1]:
                break  
            context = validation_set[:, i:i+context_length]
            ground_truth = validation_set[:, i+context_length:i+context_length+prediction_length]
            
            for s in range(context.shape[0]):
                contexts.append(context[s])    
                ground_truths.append(ground_truth[s]) 
        
    if flag=="test":
        contexts,ground_truths =[],[]
        for i in range(0, data.shape[1], stride):
            if i+context_length + prediction_length > data.shape[1]:
                break  
            context = data[:, i:i+context_length]
            ground_truth = data[:, i+context_length:i+context_length+prediction_length]
            
            for s in range(context.shape[0]):
                contexts.append(context[s])    
                ground_truths.append(ground_truth[s]) 
        
    context_tensor = torch.stack(contexts)
    ground_truth_tensor = torch.stack(ground_truths)
    
    return torch_data.TensorDataset(context_tensor, ground_truth_tensor)

def evaluate(model, val_dataset, prediction_length, context_length):
    model.eval()
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    total_loss = 0
    with torch.no_grad():
        for context, ground_truth in val_loader:
            loss = -model(context).log_prob(ground_truth).mean()
            total_loss += loss.item() * context.size(0)
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss