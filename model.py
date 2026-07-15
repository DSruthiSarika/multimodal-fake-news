"""
model.py
========
Multimodal fake-news classifier.

Architecture
------------
Text branch  : BERT-base-uncased  →  768-d CLS embedding
Image branch : ResNet-50          → 2048-d global average pool
Fusion       : concat (2816-d) → Linear(512) → ReLU → Dropout
                               → Linear(256) → ReLU → Dropout
                               → Linear(2)   (logits)
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models


class MultimodalFakeNewsClassifier(nn.Module):
    """
    Args:
        bert_model_name    : HuggingFace model id  (default: bert-base-uncased)
        fusion_dim         : hidden size of the MLP fusion head
        num_classes        : 2 for binary (real / fake)
        dropout            : dropout probability in fusion head
        freeze_bert_layers : number of BERT encoder layers to freeze (0 = fine-tune all)
        freeze_resnet      : if True, freeze all ResNet layers except layer4
    """

    def __init__(
        self,
        bert_model_name:    str  = "bert-base-uncased",
        fusion_dim:         int  = 512,
        num_classes:        int  = 2,
        dropout:            float = 0.3,
        freeze_bert_layers: int  = 8,
        freeze_resnet:      bool = True,
    ):
        super().__init__()

        # ── Text encoder (BERT) ───────────────────────────────────────────────
        self.bert     = BertModel.from_pretrained(bert_model_name)
        bert_out_dim  = self.bert.config.hidden_size   # 768

        # Freeze embeddings always
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Optionally freeze early encoder layers
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_bert_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # ── Image encoder (ResNet-50) ─────────────────────────────────────────
        resnet         = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet_out_dim = resnet.fc.in_features         # 2048

        # Drop the classification head; keep feature extractor + global pool
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # → [B, 2048, 1, 1]

        if freeze_resnet:
            # Freeze everything except layer4 (fine-tune top block only)
            for name, param in self.image_encoder.named_parameters():
                param.requires_grad = name.startswith("7.")  # layer4 is child index 7

        # ── Fusion MLP ────────────────────────────────────────────────────────
        combined_dim = bert_out_dim + resnet_out_dim   # 2816

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, max_len]
        attention_mask: torch.Tensor,   # [B, max_len]
        images:         torch.Tensor,   # [B, 3, H, W]
    ) -> torch.Tensor:                  # [B, num_classes]

        # Text features — CLS token from last hidden state (index 0)
        bert_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_out.last_hidden_state[:, 0, :]       # [B, 768]

        # Image features
        img_feat  = self.image_encoder(images)                # [B, 2048, 1, 1]
        img_feat  = img_feat.flatten(start_dim=1)             # [B, 2048]

        # Late fusion
        combined  = torch.cat([text_feat, img_feat], dim=1)   # [B, 2816]
        logits    = self.fusion(combined)                      # [B, 2]

        return logits


def count_parameters(model: nn.Module) -> dict:
    """Returns total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
