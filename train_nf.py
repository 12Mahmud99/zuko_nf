from utils import evaluate, two_moons, dataset, extraxct_data
import torch.utils.data as data
import torch
import zuko            
from tqdm import tqdm
import logging 

def train(train_dataset, val_dataset, prediction_length=50, context_length=50, batch_size=128, epochs=1000, path="best_model.pth"):
    flow = zuko.flows.NSF(features=prediction_length, transforms=3, context=context_length, hidden_features=(64, 64))
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    patience = 10
    counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        flow.train()
        total_loss = 0
        for context, ground_truth in train_loader:
            optimizer.zero_grad()
            loss = -flow(context).log_prob(ground_truth).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * context.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if epoch % 10 == 0:
            val_loss = evaluate(flow, val_dataset, prediction_length, context_length)
            print(f"Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(flow.state_dict(), path)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_length", "-p", type=int, default=50)
    parser.add_argument("--context_length","-c", type=int, default=50)
    parser.add_argument("--batch_size","-b", type=int, default=128)
    parser.add_argument("--epochs","-e", type=int, default=1000)
    parser.add_argument("--data_path","-d", type=str, required=True)
    parser.add_argument("--output","-o" , type=str, default="best_model.pth")
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()
    extracted_data = extraxct_data(args.data_path).float()
    train_dataset = dataset(flag="train", prediction_length=args.prediction_length, context_length=args.context_length, data=extracted_data)
    val_dataset = dataset(flag="val", prediction_length=args.prediction_length, context_length=args.context_length, data=extracted_data)
    train(train_dataset, val_dataset, prediction_length=args.prediction_length, context_length=args.context_length, batch_size=args.batch_size, epochs=args.epochs, path=args.output)