from .smt import SMT
from .mymodel import MyModel


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    if model_type == 'smt':
        model = SMT(img_size=config.DATA.IMG_SIZE,
                    in_chans=config.MODEL.SMT.IN_CHANS,
                    num_classes=config.MODEL.NUM_CLASSES,
                    embed_dims=config.MODEL.SMT.EMBED_DIMS,
                    ca_num_heads=config.MODEL.SMT.CA_NUM_HEADS,
                    sa_num_heads=config.MODEL.SMT.SA_NUM_HEADS,
                    mlp_ratios=config.MODEL.SMT.MLP_RATIOS,
                    qkv_bias=config.MODEL.SMT.QKV_BIAS,
                    qk_scale=config.MODEL.SMT.QK_SCALE,
                    use_layerscale=config.MODEL.SMT.USE_LAYERSCALE,
                    depths=config.MODEL.SMT.DEPTHS,
                    ca_attentions=config.MODEL.SMT.CA_ATTENTIONS,
                    head_conv=config.MODEL.SMT.HEAD_CONV,
                    expand_ratio=config.MODEL.SMT.EXPAND_RATIO,
                    drop_rate=config.MODEL.DROP_RATE,
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'mymodel':
        model = MyModel(img_size=config.DATA.IMG_SIZE,
                    in_chans=config.MODEL.SMT.IN_CHANS,
                    num_classes=config.MODEL.NUM_CLASSES,
                    num_coarses=config.MODEL.TOT.NUM_COARSES,
                    num_vocab=config.MODEL.TOT.NUM_VOCAB,
                    embed_dims=config.MODEL.SMT.EMBED_DIMS,
                    patch_sizes=config.MODEL.TOT.PATCH_SIZES,
                    paddings=config.MODEL.TOT.PADDINGS,
                    ca_num_heads=config.MODEL.SMT.CA_NUM_HEADS,
                    sa_num_heads=config.MODEL.SMT.SA_NUM_HEADS,
                    mlp_ratios=config.MODEL.SMT.MLP_RATIOS,
                    qkv_bias=config.MODEL.SMT.QKV_BIAS,
                    qk_scale=config.MODEL.SMT.QK_SCALE,
                    use_layerscale=config.MODEL.SMT.USE_LAYERSCALE,
                    depths=config.MODEL.SMT.DEPTHS,
                    ca_attentions=config.MODEL.SMT.CA_ATTENTIONS,
                    expand_ratio=config.MODEL.SMT.EXPAND_RATIO,
                    head_conv=config.MODEL.SMT.HEAD_CONV,
                    drop_rate=config.MODEL.DROP_RATE,
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
