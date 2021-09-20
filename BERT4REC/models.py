import pytorch_lightning as pl 
import torch

from loss import masked_cross_entropy 
from metrics import masked_accuracy, masked_recall_at_k


class BERT4REC(pl.LightningModule):
    def __init__(self, total_items, emb_dims, num_heads, dropout_rate, learning_rate):
        """[summary]

        Args:
            total_items ([type]): [description]
            emb_dims ([type]): [description]
            num_heads ([type]): [description]
            mask ([type]): [description]
            dropout_rate ([type]): [description]
            learning_rate ([type]): [description]
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.total_items = total_items
        self.emb_dims = emb_dims
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.total_items, 
                                                embedding_dim=self.emb_dims)
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=512,
                                                embedding_dim=self.emb_dims)
        
        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(batch_first=True,
                                                                    d_model=self.emb_dims,
                                                                    nhead=self.num_heads,
                                                                    dropout=self.dropout_rate)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, 
                                                            num_layers=2)
        self.out = torch.nn.Linear(in_features=self.emb_dims,
                                out_features=self.total_items)

    def forward(self, input):
        item_embed = self.item_embeddings(input) # 아이템 임베딩 # (batch_size, seq_len, emb_dims)

        batch_size = input.size(0)
        seq_len = input.size(1) 

        positional_encoder = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size,1)
        pos_embed = self.pos_embeddings(positional_encoder) # 포지셔녈 임베딩 (batch_size, seq_len, emb_dims)
        embed = item_embed + pos_embed

        encoded = self.transformer_encoder(embed) # batch first ( batch_size, seq_len, emb_dims)
        out = self.out(encoded) 
        return out 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                        T_0=10,
                                                                        T_mult=1,
                                                                        eta_min=1e-3,
                                                                        last_epoch=-1)
        return {
            'optimizer'     : optimizer,
            'lr_scheduler'  : scheduler, 
            'monitor'       : 'valid_loss'
        }


    def training_step(self, batch, batch_idx):
        inputs, labels, masks = batch 

        preds = self(inputs)
        preds = preds.view(-1, preds.size(2))
        
        labels = labels.view(-1)
        masks = masks.view(-1)

        loss = masked_cross_entropy(preds, labels, masks)
        acc = masked_accuracy(preds, labels, masks)
        rec = masked_recall_at_k(preds, labels, masks, 10)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log(f'train_recall@{10}', rec)

        return loss 

    def validation_step(self, batch, batch_idx):
        inputs, labels, masks = batch 

        preds = self(inputs)
        preds = preds.view(-1, preds.size(2))
        
        labels = labels.view(-1)
        masks = masks.view(-1)

        loss = masked_cross_entropy(preds, labels, masks)
        acc = masked_accuracy(preds, labels, masks)
        rec = masked_recall_at_k(preds, labels, masks, 10)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log(f'val_recall@{10}', rec)

        return loss 

    def test_step(self, batch, batch_idx):
        inputs, labels, masks = batch 

        preds = self(inputs)
        preds = preds.view(-1, preds.size(2))
        
        labels = labels.view(-1)
        masks = masks.view(-1)

        loss = masked_cross_entropy(preds, labels, masks)
        acc = masked_accuracy(preds, labels, masks)
        rec = masked_recall_at_k(preds, labels, masks, 10)

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log(f'test_recall@{10}', rec)

        return loss 