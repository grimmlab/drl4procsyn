from torch import nn


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, normalization='batch'):
        super().__init__()
        self.fn = fn
        self.norm = Normalization(dim, normalization)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class PreNormResidualTransposed(PreNormResidual):
    def forward(self, x):
        return self.fn(self.norm(x).permute((0, 2, 1))).permute((0, 2, 1)).contiguous() + x


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d,
            'layer': nn.LayerNorm
        }.get(normalization, None)

        if normalization != "layer":
            self.normalizer = normalizer_class(embed_dim, affine=True)
        else:
            self.normalizer = normalizer_class(embed_dim)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


def MixerBlock(latent_dim, num_tokens, expansion_factor=4, expansion_factor_token_mixing=2, dropout=0.0,
               normalization="batch"):
    """
    MLP-Mixer block without masking (all tokens considered) and MLP token mixing.
    Input is expected as (batch, num tokens, latent dim)
    """
    return nn.Sequential(
        PreNormResidualTransposed(latent_dim, FeedForward(num_tokens, expansion_factor_token_mixing, dropout),
                                  normalization),
        PreNormResidual(latent_dim, FeedForward(latent_dim, expansion_factor, dropout),
                        normalization)
    )


class MixerBlockFinish(nn.Module):
    """
    Mixer Block with a final normalization layer, returning a global average over the tokens
    and the transformed tokens. Can be used for various downstream tasks obtained from a shared MLP-Mixer part.
    """
    def __init__(self, latent_dim, num_tokens, expansion_factor=4, expansion_factor_token_mixing=2, dropout=0.0,
                 normalization="batch"):
        super().__init__()
        self.mixer_block = MixerBlock(latent_dim, num_tokens, expansion_factor, expansion_factor_token_mixing, dropout,
               normalization)

        self.norm = Normalization(latent_dim, normalization)

        self.projection_linear = nn.Linear(num_tokens, 1)  # projects sequence (b, latent, num_tokens) -> (b, latent, 1)

    def forward(self, x):
        """
        x expected in (batch, num tokens, latent dim)
        """
        x = self.mixer_block(x)
        pooled_token = self.projection_linear(x.permute(0, 2, 1)).squeeze(-1)
        return x, pooled_token