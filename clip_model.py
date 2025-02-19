import torch
from torch import nn
device = torch.device('cuda:0')

class CLIPClf(nn.Module):
    def __init__(self, clip, num_labels):
        super(CLIPClf, self).__init__()
        self.clip = clip
        self.num_labels = num_labels
        # Classifier head
        self.classifier = (
            nn.Linear(self.clip.ln_final.normalized_shape[0]*2, self.num_labels)
        )

    def forward(self, memes, ocr):
        memes = self.clip.encode_image(memes)
        ocr = self.clip.encode_text(ocr).float().to(device)
        outputs = torch.cat([memes, ocr], axis=1)
        outputs.to(device)
        logits = self.classifier(outputs)
        return logits