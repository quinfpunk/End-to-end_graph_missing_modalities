from bert_embedding import segment_time_series, WindowEncoder, TimeSeriesBERTCLS
import torch

if __name__ == "__main__":
    # preprocess on codes data to get this format
    data = [(1, [1]), (2, [2]), (3, [3]), (4, [2435]), (5, [3467])]
    window_size = 1 

    windows = segment_time_series(data, window_size)

    print("Segmented Windows:")
    for i, window in enumerate(windows):
        print(f"Window {i}: {window}")
    
    windows_batch = []
    for window in windows:
        names = []
        values = []
        for _, tests in window:
            for test_value in tests:
                values.append(test_value)
        test_names_tensor = torch.empty(0)
        test_values_tensor = torch.tensor(values, dtype=torch.float)
        windows_batch.append((test_names_tensor, test_values_tensor))

    # define model hyperparameters
    name_embed_dim = 8
    value_dim = 4
    hidden_dim = 16
    num_layers = 2
    num_heads = 2
    max_seq_len = len(windows_batch) + 1  # ensure this is large enough for your sequences
    
    # instantiate the model
    model = TimeSeriesBERTCLS(0, 0, value_dim, hidden_dim, 
                           num_layers, num_heads, max_seq_len)
    
    window_embeddings = []
    for test_names, test_values in windows_batch:
        print(test_names, test_values)
        emb = model.window_encoder(test_names, test_values)
        window_embeddings.append(emb)

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
    # perform a forward pass.
    output = model(window_embeddings)  # expected shape: (seq_len, hidden_dim)
    # TODO: store in a file
    print("\nTransformer Output:")
    print(output)
