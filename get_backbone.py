from mobilenet_v2 import MobileNetV2

def get_backbone(name, in_channels):
    if name == 'mobilenet_v2':
        return MobileNetV2(in_channels=in_channels)

    raise NotImplementedError
