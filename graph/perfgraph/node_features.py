INT_INIT = 0
STRING_INIT = ""


class Args:
    def __init__(self):
        self.conv_kernel_size = INT_INIT
        self.conv_stride = INT_INIT
        self.conv_padding = INT_INIT
        self.conv_dilation = INT_INIT
        self.conv_groups = INT_INIT
        self.conv_bias = INT_INIT

        self.linear_in_features = INT_INIT
        self.linear_out_features = INT_INIT
        self.linear_bias = INT_INIT

        self.pool_kernel_size = INT_INIT
        self.pool_stride = INT_INIT
        self.pool_padding = INT_INIT
        self.pool_ceil_mode = INT_INIT

    def to_list(self):
        return list(vars(self).values())


class MemoryInfo:
    def __init__(self):
        self.bytes = INT_INIT
        self.weight_size = INT_INIT
        self.batch_size = INT_INIT
        self.input_size_with_weight = INT_INIT
        self.input_size = INT_INIT
        self.input_channels = INT_INIT
        self.input_w = INT_INIT
        self.input_h = INT_INIT
        self.output_size = INT_INIT
        self.output_channels = INT_INIT
        self.output_w = INT_INIT
        self.output_h = INT_INIT

    def to_list(self):
        return list(vars(self).values())


class Feature:
    def __init__(self):
        self.type = STRING_INIT
        self.args = Args()

        self.memory_info = MemoryInfo()
        self.flops = INT_INIT
        self.arith_intensity = INT_INIT
