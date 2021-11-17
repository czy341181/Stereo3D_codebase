from lib.models.Stereo import StereoNet

def build_model(cfg):
    if cfg['type'] == 'Stereo':
        return StereoNet(cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])


