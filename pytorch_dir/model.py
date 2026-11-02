import itertools, csv
from typing import Dict, Optional, List, Tuple
import torch
from torch import nn

from dataset import wavefrom_to_mel_spectrogram, get_audio_waveform
from params import Params

def get_conv_pad_module(padding, stride, kernel_size):
    if padding == "same":
        if stride == 2:
            return nn.ZeroPad2d(padding=(0, 1, 0, 1)) 
        return nn.ZeroPad2d(padding=[(k//2) for k in kernel_size for _ in range(2)])
    return nn.Identity()

def _conv(kernel_size, stride, out_channels, in_channels, padding, eps, scale, center):
    return nn.Sequential(
        get_conv_pad_module(padding, stride, kernel_size),
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=0,
            bias=False
        ),
        # batchnorm does not have a setting for bias only like use in the tf implementation
        # we freeze the weights to 1 when loading the weights
        nn.BatchNorm2d(
            num_features=out_channels, eps=eps,
            affine=scale or center, bias=center, momentum=0.01 # tf default of 0.99
        ),
        nn.ReLU(inplace=True)
    )

def _separable_conv(kernel_size, stride, out_channels, in_channels, padding, eps, scale, center):
    padding_value = [(k//2 if padding == "same" else 0) for k in kernel_size]
    return nn.Sequential(
        # Depthwise seperable convolution
        get_conv_pad_module(padding, stride, kernel_size),
        nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, stride=stride, padding=0,
            bias=False, groups=in_channels
        ),
        nn.BatchNorm2d(
            num_features=in_channels, eps=eps,
            affine=scale or center, track_running_stats=center
        ),
        nn.ReLU(inplace=True),

        # Pointwise convolution
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0,
            bias=False, groups=1
        ),
        nn.BatchNorm2d(
            num_features=out_channels, eps=eps,
            affine=scale or center, track_running_stats=center
        ),
        nn.ReLU(inplace=True)
    )



_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,          [3, 3], 2,   32),
    (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128),
    (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256),
    (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024),
    (_separable_conv, [3, 3], 1, 1024)
]


class YamNet(nn.Module):
    """YamNet"""

    def __init__(self, params:Params):
        super().__init__()
        self.params = params

        layers = list()
        in_channels = 1
        for layer_fun, kernel_size, stride, num_filters in  _YAMNET_LAYER_DEFS:
            layers.append(layer_fun(
                kernel_size=kernel_size, stride=stride, out_channels=num_filters,
                in_channels=in_channels, padding=params.conv_padding, eps=params.batchnorm_epsilon,
                scale=params.batchnorm_scale, center=params.batchnorm_center

            ))
            in_channels = num_filters
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(in_features=_YAMNET_LAYER_DEFS[-1][-1], out_features=params.num_classes)


    def forward(
        self,
        waveform:Optional[torch.Tensor]=None,
        id:Optional[str]=None,
        start_sec:Optional[float]=None,
        end_sec:Optional[float]=None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the prediction with either the extracted waveform or the file/youtube id.
        """
        if waveform is None:
            waveform = get_audio_waveform(id, start_sec, end_sec, sr=self.params.sample_rate)
            waveform = torch.from_numpy(waveform)

        waveform_log_mel = wavefrom_to_mel_spectrogram(
            waveform, window_length_sec=self.params.stft_window_seconds, hop_size_sec=self.params.stft_hop_seconds,
            frame_window_length_sec=self.params.patch_window_seconds, frame_hop_size_sec=self.params.patch_hop_seconds,
            n_mels=self.params.mel_bands, sample_rate=self.params.sample_rate, freq_min=self.params.mel_min_hz,
            freq_max=self.params.mel_max_hz, pad=True
        )["waveform_log_mel"]

        batch, num_frame, h, w = waveform_log_mel.size()
        # flattening the frames since conv can only take 4D tensors
        waveform_log_mel = waveform_log_mel.reshape(-1, 1, h, w)

        embeddings = self.feature_extractor(waveform_log_mel).squeeze((-2, -1))
        output = self.head(embeddings).view(batch, num_frame, -1)
        return {
            "waveform_log_mel": waveform_log_mel,
            "embeddings": embeddings,
            "prediction": torch.sigmoid(output)
        }


    @classmethod
    @torch.no_grad()
    def from_tf_weights(cls, tf_dir:str = "../tensorflow_dir") -> "YamNet":

        import os, sys, importlib.util

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        # yamnet.py does `import features`, so its folder still has to be importable
        sys.path.append(tf_dir)
        yamnet_params = load_module("tf_yamnet_params", os.path.join(tf_dir, "params.py"))
        yamnet_model = load_module("tf_yamnet_model", os.path.join(tf_dir, "yamnet.py"))

        tf_model = yamnet_model.yamnet_frames_model(yamnet_params.Params())
        tf_model.load_weights(os.path.join(tf_dir, "yamnet.h5"))

        torch_model = cls(Params())
        torch_model.eval()

        def copy_conv_weight(torch_weight, tf_weight, depthwise=False):
            tf_weight = tf_weight.numpy()
            # conv weight
            assert tf_weight.ndim == 4, "conv weight must have a dim of 4"
            tf_weight_reshaped = tf_weight.transpose(2, 3, 0, 1) if depthwise\
                            else tf_weight.transpose(3, 2, 0, 1)
            torch_weight.copy_(torch.from_numpy(tf_weight_reshaped))

        def copy_linear_weight(torch_weight, tf_weight):
            tf_weight = tf_weight.numpy()
            # conv weight
            assert tf_weight.ndim == 2, "linear weight must have a dim of 2"
            tf_weight_reshaped = tf_weight.transpose(1, 0)
            torch_weight.copy_(torch.from_numpy(tf_weight_reshaped))

        def copy_1d_weight(torch_weight, tf_weight):
            tf_weight = tf_weight.numpy()
            # conv weight
            assert tf_weight.ndim == 1, "weights must have a dim of 1"
            torch_weight.copy_(torch.from_numpy(tf_weight))

        def copy_batchnorm_weight(tf_name, torch_weight=None, torch_bias=None, torch_mean=None, torch_var=None):
            weights = tf_model.get_layer(tf_name).weights
            assert len(weights) == 3, f"batch norm has to have only weights"
            if torch_weight is not None: fix_weight(torch_weight, 1.)
            if torch_bias is not None: copy_1d_weight(torch_bias, weights[0])
            if torch_mean is not None: copy_1d_weight(torch_mean, weights[1])
            if torch_var is not None: copy_1d_weight(torch_var, weights[2])

        def fix_weight(torch_weight, value):
            torch_weight.copy_(torch.full(torch_weight.size(), value))

        for name, param in itertools.chain(torch_model.named_parameters(), torch_model.named_buffers()):
            # expected torch covn name feature_extractor.<layer_index>.<module_type>.weight
            # expected torch bn name feature_extractor.<layer_index>.<module_type>.bias
            # expected tf covn name layer<layer_index+1>/(conv/depthwise_conv/pointwise_conv)
            # expected tf batch name layer<layer_index+1>/(conv/depthwise_conv/pointwise_conv)
            # .<layer_index>.<module_type>.weight/bias

            splitted_names = name.split(".")

            if len(splitted_names) == 4:
                layer_root, layer_index, layer_type, weight_type = splitted_names
                assert layer_root == "feature_extractor", \
                    f"the layer_root should be 'feature_extractor', received {layer_root}"
                layer_index = int(layer_index)
                layer_type = int(layer_type)


                # the first conv
                match layer_index:
                    case 0:
                        # the conv kernels
                        if layer_type == 1 and weight_type == "weight":
                            tf_name = f"layer{str(layer_index+1)}/conv"
                            weights = tf_model.get_layer(tf_name).weights
                            assert len(weights) == 1, f"conv has to have only weights"
                            copy_conv_weight(param, weights[0])
                        
                        # the corresponding batch norm weight fixed to 1 as it is not use in the tf implementation
                        elif layer_type == 2:
                            tf_name = f"layer{str(layer_index+1)}/conv/bn"
                            if weight_type == "weight": copy_batchnorm_weight(tf_name, torch_weight=param)
                            elif weight_type == "bias": copy_batchnorm_weight(tf_name, torch_bias=param)
                            elif weight_type == "running_mean": copy_batchnorm_weight(tf_name, torch_mean=param)
                            elif weight_type == "running_var": copy_batchnorm_weight(tf_name, torch_var=param)
                            elif weight_type == "num_batches_tracked": pass
                            else: raise RuntimeError(f"Unexpected parameter: {name}")

                        else: raise RuntimeError(f"Unexpected parameter: {name}")

                    case x if 1 <= x <= 13:
                        # Depthwise_conv or Pointwise_conv kernels
                        if (layer_type == 1 or layer_type == 4) and weight_type == "weight":
                            depthwise = layer_type == 1
                            tf_name = f"layer{str(layer_index+1)}/{('depthwise_conv' if depthwise else 'pointwise_conv')}"
                            weights = tf_model.get_layer(tf_name).weights
                            assert len(weights) == 1, f"conv has to have only weights"
                            copy_conv_weight(param, weights[0], depthwise=depthwise)

                        # the corresponding batch norm weight fixed to 1 as it is not use in the tf implementation
                        elif (layer_type == 2 or layer_type == 5):
                            depthwise = layer_type == 2
                            tf_name = f"layer{str(layer_index+1)}/{('depthwise_conv' if depthwise else 'pointwise_conv')}/bn"
                            if weight_type == "weight": copy_batchnorm_weight(tf_name, torch_weight=param)
                            elif weight_type == "bias": copy_batchnorm_weight(tf_name, torch_bias=param)
                            elif weight_type == "running_mean": copy_batchnorm_weight(tf_name, torch_mean=param)
                            elif weight_type == "running_var": copy_batchnorm_weight(tf_name, torch_var=param)
                            elif weight_type == "num_batches_tracked": pass
                            else: raise RuntimeError(f"Unexpected parameter: {name}")

                        else: raise RuntimeError(f"Unexpected parameter: {name}")

            elif len(splitted_names) == 2:
                layer_root, weight_type = splitted_names
                assert layer_root == "head", \
                    f"the layer_root should be 'head', received {layer_root}"
                tf_name = "dense"
                weights = tf_model.get_layer(tf_name).weights

                if weight_type == "weight":
                    copy_linear_weight(param, weights[0])
                elif weight_type == "bias":
                    copy_1d_weight(param, weights[1])
                else: raise RuntimeError(f"Unexpected parameter: {name}")

            else: raise RuntimeError(f"Unexpected parameter: {name}")

        return torch_model

    @classmethod
    def get_class_mapping(cls, class_map_csv) -> List[Tuple[str, str]]:
        """Read the class name definition file and return a list of strings."""
        with open(class_map_csv) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)   # Skip header
            return [(audio_set_name, display_name) for (_, audio_set_name, display_name) in reader]