import torch.nn as nn
import torch

from src.dataset.eicu_dataset import eICUDataset

class CodeEncoder():
    def __init__(self, tokenizer, embedding_size):
        super(CodeEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.code_embedding = embedding_size

        mm = torch.Tensor(self.tokenizer.code_embeddings)
        assert self.tokenizer.code_vocabs_size == mm.shape[0]
        self.code_embedding = nn.Sequential(
            nn.Embedding(self.tokenizer.code_vocabs_size, mm.shape[1]),
            # nn.Linear(mm.shape[1], embedding_size)
        )
        self.code_embedding[0].weight.data.copy_(mm)
        self.code_embedding[0].weight.requires_grad = False

    def encode(self, codes):
        # average
        codes_avg = torch.round(torch.mean(codes.type(torch.float))).to(torch.int)
        codes_emb = self.code_embedding(codes_avg)
        return codes_emb


if __name__ == '__main__':
    dataset = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")
    model = CodeEncoder(tokenizer=dataset.tokenizer, embedding_size=128)

    o = model.encode(dataset[0]["codes"])
    print(o)
    print(o.shape)
