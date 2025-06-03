from hqq.core.quantize import BaseQuantizeConfig

def get_quant_config(model):
    """
    param:
        model: the model to be quantized

    return:
        quant_config: the quant config of the model
    """
    quant_config = {}
    n_layers = model.config.num_hidden_layers
    q2_config = BaseQuantizeConfig(nbits=2, group_size=32)
    q3_config = BaseQuantizeConfig(nbits=3, group_size=128)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=32)
    q8_config = BaseQuantizeConfig(nbits=8, group_size=256)

    for i in range(n_layers):      
        if i < 4 or i > 8:
            quant_config[f'model.layers.{i}.mlp.gate_proj'] = q8_config
            quant_config[f'model.layers.{i}.mlp.up_proj'] = q8_config
            quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_config
        else:
            quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
            quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
            quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
        
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q8_config

    return quant_config