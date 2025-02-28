import torch
import torch.nn as nn
import torch.optim as optim
from embedding.bert_embedding import segment_time_series, WindowEncoder, TimeSeriesBERTCLS
from torch.utils.data import DataLoader
from src.dataset.eicu_dataset import eICUDataset
from src.dataset.utils import mimic4_collate_fn, eicu_collate_fn
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from tqdm import tqdm

class CustomTimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = self.data[idx]
        return res

def collate_time_series(data):
    print(data)
    data = torch.nn.utils.rnn.pad_sequence(
        data, batch_first=True, padding_value=0
    )
    return data

def apply_mask(window_embeddings, mask_token, mask_prob=0.15):
    seq_len, hidden_dim = window_embeddings.shape
    # Generate a Boolean mask: True means the token will be masked.
    mask = torch.rand(seq_len, device=window_embeddings.device) < mask_prob
    
    # Ensure at least one token is masked (if none, force the first one)
    if mask.sum() == 0:
        mask[0] = True
    
    # Get target embeddings for masked positions
    target_embeddings = window_embeddings[mask]  # (num_masked, hidden_dim)
    
    # Replace masked tokens with the mask token
    masked_embeddings = window_embeddings.clone()
    masked_embeddings[mask] = mask_token  # Broadcast mask_token over all masked rows
    
    return masked_embeddings, mask, target_embeddings


if __name__ == "__main__":
    # Assume TimeSeriesBertCLS is defined as in the previous code snippet.
    # Also assume that train_loader is a PyTorch DataLoader that yields batches of data.
    # Each batch is assumed to be a tuple: (windows_batch, target)
    # windows_batch: list of window tokens, each token is (test_names_tensor, test_values_tensor)
    # target: Tensor containing the ground truth for the batch (e.g., regression targets).

    collate_fn = eicu_collate_fn
    train_set = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")

    data = [] # train_set[0]["codes"]
    for i in range(len(train_set)):
        patient_data = train_set[i]
        patient_code = patient_data["lab"]
        data.append(patient_code)
    window_size = 60
    windows = []
    for d in data:
        window = segment_time_series(d, window_size)
        windows.append(window)
    # for debugging purpose, should use logging ?
    # print("Segmented Windows:")
    # for i, window in enumerate(windows):
    #     print(f"Window {i}: {window}")
    # should be done on the whole dataset
    test_names_set = set()
    for elt in data:
        if len(elt) == 0:
            continue
        for patient_value in elt:
            _, tests = patient_value
            for test_name, _ in tests:
                test_names_set.add(test_name)
    test_names_list = sorted(list(test_names_set))
    name2idx = {name: idx for idx, name in enumerate(test_names_list)}
    vocab_size = len(name2idx)

    windows_batch = []
    for w in windows:
        names = []
        values = []
        window_batch = []
        for window in w:
            for _, tests in window:
                for test_name, test_value in tests:
                    names.append(name2idx[test_name])
                    values.append(test_value)
            test_names_tensor = torch.tensor(names, dtype=torch.long)
            test_values_tensor = torch.tensor(values, dtype=torch.float)
            window_batch.append((test_names_tensor, test_values_tensor))
        windows_batch.append(window_batch)
    dataset = CustomTimeSeriesDataset(windows_batch)
    # train_loader = DataLoader(CustomTimeSeriesDataset(windows_batch), batch_size=64, num_workers=0, collate_fn=collate_time_series, shuffle=True)
    # print(next(iter(train_loader)))
    # Hyperparameters
    learning_rate = 1e-3
    num_epochs = 10

    # Instantiate your model with appropriate hyperparameters.
    # (These variables like vocab_size, name_embed_dim, etc., should be defined based on your data.)
    name_embed_dim = 8
    value_dim = 4
    hidden_dim = 16
    num_layers = 2
    num_heads = 2
    max_seq_len = len(windows_batch) + 1  # ensure this is large enough for your sequences
    model = TimeSeriesBERTCLS(vocab_size, name_embed_dim, value_dim, hidden_dim, 
                           num_layers, num_heads, max_seq_len)

    # Define the optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # For regression tasks. Use CrossEntropyLoss for classification.

    # Example training loop:
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode.
        running_loss = 0.0
        
        for batch in tqdm(windows_batch):
            window_embeddings = []
            for test_names, test_values in batch:
                emb = model.window_encoder(test_names, test_values)
                window_embeddings.append(emb)
            if len(window_embeddings) == 0:
                continue
            # optionally pad/truncate window_embeddings to self.max_seq_len
            seq_len = len(window_embeddings)
            if seq_len < model.max_seq_len:
                pad_tensor = torch.zeros(model.max_seq_len - seq_len, window_embeddings[0].shape[-1],
                                         device=window_embeddings[0].device)
                window_embeddings = window_embeddings + [pad_tensor[i] for i in range(pad_tensor.size(0))]
            else:
                window_embeddings = window_embeddings[:model.max_seq_len]
            
            # stack and prepend the CLS token: shape becomes (max_seq_len+1, hidden_dim)
            window_embeddings = torch.stack(window_embeddings, dim=0)
            # Unpack the batch.
            # windows_batch: list of windows tokens for one sample.
            # target: ground truth tensor.
            masked_embeddings, mask, target_embeddings = apply_mask(window_embeddings, model.mask_token, mask_prob=0.15)
            
            # Zero the gradients.
            
            # Forward pass.
            # The model expects windows_batch to be a list of window tokens.
            output = model(masked_embeddings)  # For TimeSeriesBertCLS, output is a fixed-size tensor.
            
            # masked_output = output[mask]
            # Compute loss.
            loss = criterion(output, target_embeddings.squeeze())
            
            optimizer.zero_grad()
            # Backward pass.
            loss.backward()
            
            # Update model parameters.
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "embedding/lab_model.pth")

