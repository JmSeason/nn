import tensorflow as tf

def mlp_create(layer_create_funtions):
    if(layer_create_funtions is None or len(layer_create_funtions) == 0):
        return None

    mlp_out = layer_create_funtions.pop(0)()  #取第一层函数并运行
    for func in layer_create_funtions:
        mlp_out = func(mlp_out)

    return mlp_out
