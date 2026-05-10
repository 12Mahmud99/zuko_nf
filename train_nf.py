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
        
    if flag=="test":
        contexts,ground_truths =[],[]
        for i in range(0, data.shape[1], stride):
            if i+context_length + prediction_length > data.shape[1]:
                break  
            context = data[:, i:i+context_length]
            ground_truth = data[:, i+context_length:i+context_length+prediction_length]
            
            contexts.append(context)
            ground_truths.append(ground_truth)
        
    context_tensor = torch.stack(contexts)
    ground_truth_tensor = torch.stack(ground_truths)
    
    return data.TensorDataset(context_tensor, ground_truth_tensor)
            

def train(dataset, prediction_length=50, context_length=50, batch_size=128, epochs=1000):
    flow = zuko.flows.NSF(features=prediction_length, transforms=3, context=context_length, hidden_features=(64, 64))
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_length", type=int, default=50)
    parser.add_argument("--context_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--data_path","-d", type=str, required=True)
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()