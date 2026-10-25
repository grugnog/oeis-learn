"""Closed hardware-only shapes and conservative reservations; no Torch on host."""
from .identity import seal

PROFILES={key:seal(dict(version='pipeline-model-size/v1',name=key,width=d,heads=h,
    encoder_layers=e,decoder_layers=l,ff=4*d,precision='float32',backend='eager',seed=20261016))
    for key,d,h,e,l in [('m8',256,4,4,4),('m32',512,8,4,4),('m100',768,12,6,6)]}
TRIALS=['cold','warmup','calibration-0','calibration-1','calibration-2']
SLOTS=[f'sizing.{key}.{trial}' for key in PROFILES for trial in TRIALS]
EXECUTION=seal(dict(version='pipeline-sizing-execution/v1',mha_fastpath=False,sdpa_backends=['math'],
    float32_matmul='highest',matmul_tf32=False,cudnn_tf32=False,deterministic_algorithms=True,
    deterministic_warn_only=False,dtype='torch.float32'))


def validate_profile(profile):
    if not isinstance(profile,dict) or profile!=PROFILES.get(profile.get('name')):
        raise ValueError('exact registered hardware-only model profile required')
    return profile


def parameters(profile):
    p=validate_profile(profile);d=p['width'];f=p['ff'];v=86
    linear=lambda a,b:a*b+b
    encoder_layer=lambda d,f:4*d*d+4*d+2*d*f+f+d+4*d
    decoder_layer=8*d*d+8*d+2*d*f+f+d+6*d
    encoder=(p['encoder_layers']*encoder_layer(d,f)+linear(1,f)+linear(f,d)+2*d
        +linear(32,d)+2*d+16*33*16+linear(258,d)+2*d
        +2*linear(d,2*d)+2*d+2*d+256*4+linear(33,64)
        +encoder_layer(64,256)+linear(132,d)+d)
    decoder=p['decoder_layers']*decoder_layer+2*d+2*v*d+v
    return dict(encoder=encoder,decoder=decoder,total=encoder+decoder,active=encoder+decoder)


def peak_estimate(profile):
    p=validate_profile(profile);d=p['width'];l=p['decoder_layers'];h=p['heads']
    # Params/gradients/AdamW/temporary state plus conservative full1024 FP32
    # training activations/attention and8GiB runtime/native/input allowance.
    return (8<<30)+32*parameters(p)['total']+8*1024*d*l*256+8*h*1024**2*8*l
