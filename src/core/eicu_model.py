import torch
import torch.nn as nn


class eICUBackbone(nn.Module):
    def __init__(
            self,
            embedding_size,
            dropout,
            num_classes,
    ):
        super(eICUBackbone, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classifier = nn.Sequential(
            nn.Linear(1208, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_classes),
        )

        self.num_classes = num_classes
        if num_classes == 1:
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Softmax(dim=-1)

    def forward(
            self,
            age,
            gender,
            ethnicity,
            codes,
            labvectors,
            apacheapsvar,
            label,
            **kwargs,
    ):
        age = age.to(self.device)
        gender = gender.to(self.device)
        ethnicity = ethnicity.to(self.device)
        codes = codes.to(self.device)
        labvectors = labvectors.to(self.device)
        apacheapsvar = apacheapsvar.to(self.device)
        label = label.to(self.device)

        inputs = torch.cat((age, gender, ethnicity, codes, labvectors, apacheapsvar), dim=1)

        logits = self.classifier(inputs)

        loss = self.classification_loss(logits, label)

        return loss

    def classification_loss(self, l, y):
        if self.num_classes == 1:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(l.squeeze(-1), y)
        else:
            loss = torch.nn.functional.cross_entropy(l, y)
        return loss

    def inference(
            self,
            age,
            gender,
            ethnicity,
            codes,
            labvectors,
            apacheapsvar,
            **kwargs,
    ):

        inputs = torch.cat((age, gender, ethnicity, codes, labvectors, apacheapsvar), dim=1)

        logits = self.classifier(inputs)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        y_scores = self.act(logits)

        return y_scores, logits
