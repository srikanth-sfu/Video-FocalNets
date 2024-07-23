import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, vit_base_patch16_224, _create_vision_transformer

class ViTVideo(nn.Module):
    def __init__(self, pretrained_vit: VisionTransformer, num_frames: int, num_classes: int, pretrained: bool):
        super(ViTVideo, self).__init__()
        self.patch_embedding = pretrained_vit.patch_embed
        self.cls_token = pretrained_vit.cls_token
        self.positional_encoding = pretrained_vit.pos_embed
        self.pretrained = pretrained
        if pretrained:
            self.transformer = pretrained_vit.blocks
            self.norm = pretrained_vit.norm
        else:
            model_args = dict(patch_size=16*num_frames, embed_dim=768, depth=12, num_heads=12)
            model = _create_vision_transformer('vit_base_patch16_224', pretrained=False, **dict(model_args))
            self.transformer = model.blocks
            self.norm = model.norm
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
        if not self.pretrained:
            x = x.view(B,-1,x.size(-1))
        x = self.transformer(x)
        
        x = self.norm(x)
        cls_tokens_final = x[:, 0]
        
        # Mean pooling of the final layer patch representations
        print(cls_tokens_final.size(), x.size())
        cls_tokens_final = cls_tokens_final.view(B, T, -1).mean(dim=1)
        
        x = self.head(cls_tokens_final)
        
        return x

def create_model(num_classes, num_frames=8, pretrained=True):
    pretrained_vit = vit_base_patch16_224(pretrained=True)
    model = ViTVideo(pretrained_vit, num_frames, num_classes, pretrained)
    
    return model

## Load pretrained ViT model
#pretrained_vit = vit_base_patch16_224(pretrained=True)
if __name__ == "__main__":
    num_frames = 8
    num_classes = 51
    model = create_model(num_classes=num_frames, num_frames=num_frames, pretrained=False)
    
    # Example usage
    dummy_input = torch.randn(2, num_frames, 3, 224, 224)  # (batch_size, num_frames, channels, height, width)
    output = model(dummy_input)
    print(output.shape)  # Should be (batch_size, num_classes)

