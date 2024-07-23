from timm.models import create_model
from . import videofocalnet
from transformers import CLIPModel, CLIPProcessor
from . import vit_base

def build_model(config):
    model_type = config.MODEL.TYPE
    is_pretrained = config.MODEL.PRETRAINED 
    print(f"Creating model: {model_type}")
    processor = None    
    if "focal" in model_type:
        model = create_model(
            model_type, 
            pretrained=is_pretrained, 
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            focal_levels=config.MODEL.FOCAL.FOCAL_LEVELS, 
            focal_windows=config.MODEL.FOCAL.FOCAL_WINDOWS, 
            use_conv_embed=config.MODEL.FOCAL.USE_CONV_EMBED, 
            use_layerscale=config.MODEL.FOCAL.USE_LAYERSCALE,
            use_postln=config.MODEL.FOCAL.USE_POSTLN, 
            use_postln_in_modulation=config.MODEL.FOCAL.USE_POSTLN_IN_MODULATION, 
            normalize_modulator=config.MODEL.FOCAL.NORMALIZE_MODULATOR,
            num_frames=config.DATA.NUM_FRAMES,
            tubelet_size=config.MODEL.TUBELET_SIZE
        )                      
    elif "vit_base" in model_type:
        model = vit_base.create_model(
            num_frames=config.DATA.NUM_FRAMES,
            num_classes=config.MODEL.NUM_CLASSES
        )
    elif "vit" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif "resnet" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=config.MODEL.NUM_CLASSES
        )
    elif "clip" in model_type:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=config.MODEL.NUM_CLASSES
        )        
    return model, processor
