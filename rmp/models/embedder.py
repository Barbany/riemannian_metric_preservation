import torch


class Embedder:
    """Positional encoding"""

    def __init__(self, input_dims, include_input, log_sampling, multires, **kwargs):
        self.input_dims = input_dims
        self.include_input = include_input
        self.periodic_functions = [torch.sin, torch.cos]
        self.log_sampling = log_sampling
        self.max_frequency_log2 = multires - 1
        self.num_frequencies = multires
        self.embed_fns, self.out_dim = self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_frequency_log2
        num_frequencies = self.num_frequencies

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=num_frequencies)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=num_frequencies)

        for freq in freq_bands:
            for p_fn in self.periodic_functions:
                embed_fns.append(lambda x: p_fn(x * freq))
                out_dim += d
        return embed_fns, out_dim

    def __call__(self, inputs):
        # Rescale inputs from [0, 1] to [-1, 1]
        return torch.cat([fn(2 * inputs - 1) for fn in self.embed_fns], -1)
