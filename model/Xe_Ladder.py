import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== GMM Prior =====
class GMMPrior(nn.Module):
    def __init__(self, latent_dim, num_components=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.means = nn.Parameter(torch.randn(num_components, latent_dim))
        self.log_vars = nn.Parameter(torch.zeros(num_components, latent_dim))
        self.weights = nn.Parameter(torch.ones(num_components) / num_components)

    def forward(self, z, mu_q, log_var_q):
        loss = 0
        weights = torch.softmax(self.weights, dim=0)
        for i in range(self.num_components):
            mean = self.means[i]
            log_var = self.log_vars[i]
            weight = weights[i]
            epsilon = 1e-6
            var = torch.exp(log_var) + epsilon
            var_q = torch.exp(log_var_q) + epsilon
            loss += weight * torch.mean(
                0.5 * (
                        torch.log(var / var_q) +
                        (var_q + (mu_q - mean).pow(2)) / var - 1.0
                )
            )
        return loss


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class ConvBlockWithCBAM(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, 3, padding=1),
            nn.ReLU()
        )
        self.cbam = CBAM(out_chans)
        self.dropout = nn.Dropout2d(drop_prob)
    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return self.dropout(x)

class LatentBlock(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.to_mu = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, latent_dim)
        )
        self.to_logvar = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, latent_dim)
        )
    def forward(self, x):
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        return mu, logvar

class LatentDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, out_size=3):
        super().__init__()
        self.linear = nn.Linear(latent_dim, out_channels * out_size * out_size)
        self.out_channels = out_channels
        self.out_size = out_size
    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, self.out_channels, self.out_size, self.out_size)
        return x


class UNetModel_Xe(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob=0.1, latent_dim=128, components=4):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        self.down_layers = nn.ModuleList()
        self.latent_blocks = nn.ModuleList()
        self.latent_decoders = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.skip_channels = []

        self.GMMPrior = GMMPrior(latent_dim, components)

        # Encoder
        ch = chans
        prev_ch = chans
        for i in range(num_pool_layers):
            in_c = in_chans if i == 0 else prev_ch
            out_c = ch
            self.down_layers.append(ConvBlockWithCBAM(in_c, out_c, drop_prob))
            self.latent_blocks.append(LatentBlock(out_c, latent_dim))
            self.skip_channels.append(out_c)
            prev_ch = out_c
            ch *= 2

        self.bottleneck = ConvBlockWithCBAM(prev_ch, ch, drop_prob)

        # Decoder
        decoder_in_ch = ch
        for i in range(num_pool_layers-1, -1, -1):
            skip_ch = self.skip_channels[i]
            z_feat_ch = skip_ch
            in_c = decoder_in_ch + skip_ch + z_feat_ch
            out_c = skip_ch
            self.latent_decoders.append(LatentDecoder(latent_dim, z_feat_ch, out_size=3))
            self.up_layers.append(ConvBlockWithCBAM(in_c, out_c, drop_prob))
            decoder_in_ch = out_c

        self.final_conv = nn.Conv2d(decoder_in_ch, out_chans, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        skips, latents, zs, spatial_sizes = [], [], [], []
        out = x

        for i in range(self.num_pool_layers):
            out = self.down_layers[i](out)
            skips.append(out)
            spatial_sizes.append(out.shape[-2:])
            mu, logvar = self.latent_blocks[i](out)
            z = self.reparameterize(mu, logvar)
            zs.append(z)
            latents.append((mu, logvar))
            out = self.pool(out)

        out = self.bottleneck(out)

        for i in range(self.num_pool_layers-1, -1, -1):
            target_size = spatial_sizes[i]
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            z_feat = self.latent_decoders[self.num_pool_layers-1-i](zs[i])
            z_feat = F.interpolate(z_feat, size=target_size, mode='bilinear', align_corners=False)
            out = torch.cat([out, skips[i], z_feat], dim=1)
            out = self.up_layers[self.num_pool_layers-1-i](out)

        out = self.final_conv(out)
        prior_loss = sum(self.GMMPrior(z, mu, logvar) for z, (mu, logvar) in zip(zs, latents))
        return out, prior_loss
