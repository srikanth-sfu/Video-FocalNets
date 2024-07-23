import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, vit_base_patch16_224

class ViTVideo(nn.Module):
    def __init__(self, pretrained_vit: VisionTransformer, num_frames: int, num_classes: int):
        super(ViTVideo, self).__init__()
        self.patch_embedding = pretrained_vit.patch_embed
        self.cls_token = pretrained_vit.cls_token
        self.positional_encoding = pretrained_vit.pos_embed
        self.transformer = pretrained_vit.blocks
        self.norm = pretrained_vit.norm
        self.head = nn.Linear(pretrained_vit.embed_dim, num_classes)

        # Adjust positional encoding to handle multiple frames
        self.frame_positional_encoding = nn.Parameter(
            torch.zeros(1, num_frames, pretrained_vit.embed_dim)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        x = self.patch_embedding(x)  # (B*T, num_patches, embed_dim)
        
        x = x.view(B, T, -1, x.size(-1))  # (B, T, num_patches, embed_dim)
        x = x + self.frame_positional_encoding[:, :T, None, :]  # (B, T, num_patches, embed_dim)
        
        x = x.view(B * T, -1, x.size(-1))  # (B*T, num_patches, embed_dim)
        
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_encoding
        x = self.transformer(x)
        
        x = self.norm(x)
        cls_tokens_final = x[:, 0]
        
        # Mean pooling of the final layer patch representations
        cls_tokens_final = cls_tokens_final.view(B, T, -1).mean(dim=1)
        
        x = self.head(cls_tokens_final)
        
        return x

def create_model(num_classes, num_frames=8):
    pretrained_vit = vit_base_patch16_224(pretrained=True)
    model = ViTVideo(pretrained_vit, num_frames, num_classes)
    
    return model

## Load pretrained ViT model
#pretrained_vit = vit_base_patch16_224(pretrained=True)
#num_frames = 8
#num_classes = 1000
#model = ViTVideo(pretrained_vit, num_frames, num_classes)
#
## Example usage
#dummy_input = torch.randn(2, num_frames, 3, 224, 224)  # (batch_size, num_frames, channels, height, width)
#output = model(dummy_input)
#print(output.shape)  # Should be (batch_size, num_classes)

