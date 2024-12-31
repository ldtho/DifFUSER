from typing import List, Optional
import torch.nn.functional as F
import torch
from torch import nn
import math
from mmdet3d.models.builder import FUSERS
from mmcv.runner import auto_fp16, force_fp32
from .utils import cosine_beta_schedule, extract, replace_nan, check_nan, SinusoidalPositionEmbeddings
from .fuser_blocks import AttentionBlock, SeparableConvBlock, DoubleInputSequential
__all__ = ["Diffuser"]








@FUSERS.register_module()
class Diffuser(nn.Module):  # input size: batch, C, width, length [1,64,400,600]
    def __init__(self,
                 num_blocks=[3, 5, 5],
                 mid_planes=[336, 128, 256],
                 strides=[1, 2, 2],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[128, 128, 128],
                 in_channels=[256, 86],
                 out_channels=None,  # unused
                 onnx_export=False,
                 attention=True,

                 num_timesteps=1000,
                 sampling_timesteps=10,
                 objective='pred_x0',

                 sampler='ddim',  # 'ddim' 'dpmsolver' 'dpmsolver++' "sde-dpmsolver" "sde-dpmsolver++"
                 # FOR DPM SOLVER
                 solver_type='midpoint',  # for dpm solver
                 lower_order_final=True,
                 solver_order=2,

                 # FOR DDIM SOLVER
                 ddim_sampling_eta=0.0,

                 # ablation
                 shift_scale_gate='shift_scale_gate',
                 use_diffusion_loss = True,

                 sensor_mask_prob=0.0,
                 sensor_mask='random',
                 return_intermediate=False,
                 ):
        super(Diffuser, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_planes = sum(in_channels)
        self.act = nn.SiLU()
        self.layer1 = []
        self.sensor_mask_prob = sensor_mask_prob
        self.sensor_mask = sensor_mask
        self.return_intermediate = return_intermediate
        self.use_diffusion_loss = use_diffusion_loss
        self.num_upsample_filters = num_upsample_filters
        self.out_channels = out_channels

        for i in range(num_blocks[0]):
            self.layer1.append(AttentionBlock(in_channels=self.in_planes if i == 0 else mid_planes[0],
                                              out_channels=mid_planes[0],
                                              stride=strides[0] if i == 0 else 1,
                                              shift_scale_gate=shift_scale_gate,
                                              # first_layer=i == 0))
                                              first_layer=False))
        self.layer1 = DoubleInputSequential(*self.layer1)
        upsample_strides = list(map(float, upsample_strides))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[0],
                               int(upsample_strides[0]),
                               stride=int(upsample_strides[0]), bias=False
                               ) if upsample_strides[0] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[0],
                      kernel_size=int(1 / upsample_strides[0]),
                      stride=int(1 / upsample_strides[0]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.1),
            self.act
        )
        self.layer2 = []
        for i in range(num_blocks[1]):
            self.layer2.append(AttentionBlock(in_channels=mid_planes[0] if i == 0 else mid_planes[1],
                                              out_channels=mid_planes[1],
                                              stride=strides[1] if i == 0 else 1,
                                              shift_scale_gate=shift_scale_gate,
                                              first_layer=i == 0))
        self.layer2 = DoubleInputSequential(*self.layer2)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[1],
                               int(upsample_strides[1]),
                               stride=int(upsample_strides[1]), bias=False
                               ) if upsample_strides[1] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[1],
                      kernel_size=int(1 / upsample_strides[1]),
                      stride=int(1 / upsample_strides[1]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[1], eps=1e-3, momentum=0.1),
            self.act
        )

        self.layer3 = []
        for i in range(num_blocks[2]):
            self.layer3.append(AttentionBlock(in_channels=mid_planes[1] if i == 0 else mid_planes[2],
                                              out_channels=mid_planes[2],
                                              stride=strides[2] if i == 0 else 1,
                                              shift_scale_gate=shift_scale_gate,
                                              first_layer=i == 0))
        self.layer3 = DoubleInputSequential(*self.layer3)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[2],
                               int(upsample_strides[2]),
                               stride=int(upsample_strides[2]), bias=False
                               ) if upsample_strides[2] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[2],
                      kernel_size=int(1 / upsample_strides[2]),
                      stride=int(1 / upsample_strides[2]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[2], eps=1e-3, momentum=0.1),
            self.act
        )

        self.p1_down_channel = nn.Sequential(nn.Conv2d(mid_planes[0], mid_planes[1], 3, padding=1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.1, eps=1e-3),
                                             )
        self.p2_down_channel = nn.Sequential(nn.Conv2d(mid_planes[1], mid_planes[1], 3, padding=1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.1, eps=1e-3),
                                             )
        self.p3_down_channel = nn.Sequential(nn.Conv2d(mid_planes[2], mid_planes[1], 3, padding=1),
                                             nn.BatchNorm2d(mid_planes[1], momentum=0.1, eps=1e-3),
                                             )
        self.conv2_up = SeparableConvBlock(mid_planes[1], norm=True, activation=False, onnx_export=onnx_export)
        self.conv1_up = SeparableConvBlock(mid_planes[1], norm=True, activation=True,
                                           onnx_export=onnx_export)  # use act

        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = nn.MaxPool2d((2, 2))
        self.p3_downsample = nn.MaxPool2d((2, 2))

        self.conv2_down = SeparableConvBlock(mid_planes[1], norm=True, activation=True,
                                             onnx_export=onnx_export)  # use act
        self.conv3_down = SeparableConvBlock(mid_planes[1], norm=True, activation=True,
                                             onnx_export=onnx_export)  # use act
        self.conv_last = nn.Conv2d(sum(num_upsample_filters), self.in_planes, kernel_size=1)
        self.conv_out = nn.Conv2d(sum(num_upsample_filters), self.in_planes, kernel_size=1)
        self.relu = nn.SiLU()
        self.attention = attention
        if attention:
            self.epsilon = 1e-5
            self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p3_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # Diffusion
        self.sampler = sampler
        self.num_timesteps = num_timesteps
        self.sampling_timesteps = sampling_timesteps
        print("sampling_timesteps", self.sampling_timesteps)
        self.objective = objective
        self.ddim_sampling_eta = ddim_sampling_eta

        betas = cosine_beta_schedule(self.num_timesteps)
        # betas = self.enforce_zero_terminal_snr(betas)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps = betas.shape

        assert self.sampling_timesteps <= self.num_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.self_condition = False

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # DPM
        sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        lambda_t = torch.log(torch.sqrt(alphas_cumprod)) - torch.log(sigma_t)
        self.register_buffer('sigma_t', sigma_t)
        self.register_buffer('lambda_t', lambda_t)
        self.model_outputs = [None] * solver_order
        self.lower_order_final = lower_order_final
        self.solver_order = solver_order
        self.solver_type = solver_type
        self.lower_order_nums = 0

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(self.in_planes),
            nn.Linear(self.in_planes, self.in_planes * 4, bias=True),
            nn.SiLU(),
            nn.Linear(self.in_planes * 4, self.in_planes, bias=True),
        )

        self.init_weights(weight_init='xavier')
        self.to(self.device)

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # batchnorm
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward(self, p1, p2, p3):
        p2_up = self.conv2_up(self.act(p2 + self.p2_upsample(p3)))
        p1_out = self.conv1_up(self.act(p1 + self.p1_upsample(p2_up)))
        p2_out = self.conv2_down(self.act(p2 + p2_up + self.p2_downsample(p1_out)))
        p3_out = self.conv3_down(self.act(p3 + self.p3_downsample(p2_out)))

        return p1_out, p2_out, p3_out

    def _forward_fast_attention(self, p1, p2, p3):
        p2_w1 = self.relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_up = self.conv2_up(self.act(weight[0] * p2 + weight[1] * self.p2_upsample(p3)))
        p1_w1 = self.relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)

        p1_out = self.conv1_up(self.act(weight[0] * p1 + weight[1] * self.p1_upsample(p2_up)))

        p2_w2 = self.relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p2_out = self.conv2_down(self.act(weight[0] * p2 + weight[1] * p2_up + weight[2] * self.p2_downsample(p1_out)))

        p3_w2 = self.relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv3_down(self.act(weight[0] * p3 + weight[1] * self.p3_downsample(p2_out)))

        return p1_out, p2_out, p3_out

    @force_fp32(apply_to=('pred',))
    def get_noise_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def forward(self, x_start):
        feat_channels = [0] + [x.shape[1] for x in x_start]
        feat_channels = torch.cumsum(torch.tensor(feat_channels), dim=0)
        x_clean = torch.cat(x_start, dim=1)
        noise = torch.clamp(torch.randn_like(x_clean, device=self.device), min=-2, max=2)
        target = x_clean.detach().clone() if self.objective == 'pred_x0' else noise

        # x_start = torch.cat(x_start, dim=1)
        if self.training:
            t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        else:
            t = torch.tensor([999], device=self.device).long()

        if self.sensor_mask_prob > 0.0:
            sensor_mask = torch.rand(1, device=self.device) < self.sensor_mask_prob
            x_t = torch.cat(x_start, dim=1)
            condition = torch.cat(x_start, dim=1)
            if sensor_mask:
                if self.sensor_mask == 'random':
                    # idx = torch.randint(0, 2, (1,), device=self.device).long()
                    idx = torch.randint(1, 3, (1,), device=self.device).long()
                elif self.sensor_mask == 'lidar':
                    idx = torch.tensor([0], device=self.device).long()
                elif self.sensor_mask == 'camera':
                    idx = torch.tensor([1], device=self.device).long()
                if not (self.sensor_mask == 'random' and not self.training):
                    condition[:, feat_channels[idx-1]:feat_channels[idx], :, :] = condition[:, feat_channels[idx-1]:feat_channels[idx], :, :] * 0.0
            x_t = self.q_sample(x_t, t, noise=noise)
        else:
            condition = torch.cat(x_start, dim=1)
            x_t = self.q_sample(condition, t, noise=noise)


        diffuse_loss = 0
        intermediate_feats = []

        if self.training:
            x_pred, noise_pred = self.model_prediction(x_t, condition, t)
            if self.objective == 'pred_noise':
                diffuse_loss = self.get_noise_loss(noise_pred, target)
            elif self.objective == 'pred_x0':
                diffuse_loss = self.get_noise_loss(x_pred, target)

        else:
            if self.sampler == 'ddim':
                x_pred, noise_pred, intermediate_feats = self.ddim_sampling(x_t, condition)
            elif self.sampler in ['dpmsolver', 'dpmsolver++', "sde-dpmsolver", "sde-dpmsolver++", 'deis']:
                x_pred, noise_pred, intermediate_feats = self.dpm_deis_sampling(x_t, condition)
            else:
                raise NotImplementedError

        COUNT_FLOPS = False
        if COUNT_FLOPS:
            return x_pred

        return x_pred, diffuse_loss if self.use_diffusion_loss else torch.tensor(0.0, device=self.device), intermediate_feats

    def model_prediction(self, x_t, x_start, ts):
        t = self.time_emb(ts)
        # Ver 1
        c = x_start + t.unsqueeze(-1).unsqueeze(-1)

        x, c1 = self.layer1(x_t, c)
        p1 = self.p1_down_channel(x)

        x, c2 = self.layer2(x, c1)
        p2 = self.p2_down_channel(x)

        x, c3 = self.layer3(x, c2)
        p3 = self.p3_down_channel(x)

        if self.attention:
            p1_out, p2_out, p3_out = self._forward_fast_attention(p1, p2, p3)
        else:
            p1_out, p2_out, p3_out = self._forward(p1, p2, p3)

        up1 = self.deconv1(p1_out)
        up2 = self.deconv2(p2_out)
        up3 = self.deconv3(p3_out)

        x = F.silu(torch.cat([up1, up2, up3], dim=1))

        if self.objective == 'pred_x0':
            x_pred = self.conv_last(x)
            pred_noise = self.predict_noise_from_start(x_start,
                                                       ts,
                                                       x_pred)
        elif self.objective == 'pred_noise':
            pred_noise = self.conv_last(x)
            x_pred = self.predict_start_from_noise(x_start,
                                                   ts,
                                                   pred_noise)

        return x_pred, pred_noise

    @torch.no_grad()
    def ddim_sampling(self, x_t, x_start):
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        intermediate_feats = []
        for i, (time, time_next) in enumerate(time_pairs):
            ts = torch.tensor([time], device=x_start.device, dtype=torch.long)
            x_t, pred_noise = self.model_prediction(x_t, x_start, ts)
            if time_next < 0:
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x_t)
            x_t = x_t * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if self.return_intermediate:
                intermediate_feats.append(x_t)
        return x_t, pred_noise, intermediate_feats

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        ).float()

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        ).float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).float()

    def dpm_deis_sampling(self, x_t, x_start, return_intermediate = False):
        total_timesteps, sampling_timesteps, objective = self.num_timesteps, self.sampling_timesteps, self.objective
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        self.timesteps = times[:-1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        intermediate_feats = []
        for step_index, (time, time_next) in enumerate(time_pairs):
            time_next = max(time_next, 0)
            if time == time_next == 0:
                continue
            ts = torch.tensor([time], device=x_start.device, dtype=torch.long)
            x_t, pred_noise = self.model_prediction(x_t, x_start, ts)
            lower_order_final = (
                    (step_index == len(time_pairs) - 1) and self.lower_order_final and self.sampling_timesteps <= 15
            )
            lower_order_second = (
                    (step_index == len(time_pairs) - 2) and self.lower_order_final and self.sampling_timesteps <= 15
            )
            for i in range(self.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
            self.model_outputs[-1] = x_t
            if self.sampler in ["sde-dpmsolver", "sde-dpmsolver++"]:
                noise = torch.randn_like(x_t)
            else:
                noise = None

            if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
                if 'dpm' in self.sampler:
                    x_t = self.dpm_solver_first_order_update(
                        x_t, time, time_next, x_t, noise=noise
                    )
                elif 'deis' in self.sampler:
                    x_t = self.deis_first_order_update(
                        x_t, time, time_next, x_t)
                else:
                    raise NotImplementedError
            elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
                timestep_list = [self.timesteps[step_index - 1], time]
                if 'dpm' in self.sampler:

                    x_t = self.multistep_dpm_solver_second_order_update(
                        self.model_outputs, timestep_list, time_next, x_t, noise=noise
                    )
                elif 'deis' in self.sampler:
                    x_t = self.multistep_deis_second_order_update(
                        self.model_outputs, timestep_list, time_next, x_t
                    )
                else:
                    raise NotImplementedError
            else:
                timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], time]
                if 'dpm' in self.sampler:
                    x_t = self.multistep_dpm_solver_third_order_update(
                        self.model_outputs, timestep_list, time_next, x_t, noise=noise
                    )
                elif 'deis' in self.sampler:
                    x_t = self.multistep_deis_third_order_update(
                        self.model_outputs, timestep_list, time_next, x_t
                    )
                else:
                    raise NotImplementedError
            if self.return_intermediate:
                intermediate_feats.append(x_t)
            if self.lower_order_nums < self.solver_order:
                self.lower_order_nums += 1

        return x_t, pred_noise, intermediate_feats

    def deis_first_order_update(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        One step for the first-order DEIS (equivalent to DDIM).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.sqrt_alphas_cumprod[prev_timestep], self.sqrt_alphas_cumprod[timestep]
        sigma_t, _ = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        if self.sampler == "deis":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        else:
            raise NotImplementedError("only support log-rho multistep deis now")
        return x_t

    def dpm_solver_first_order_update(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            prev_timestep: int,
            sample: torch.FloatTensor,
            noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        One step for the first-order DPM-Solver (equivalent to DDIM).

        See https://arxiv.org/abs/2206.00927 for the detailed derivation.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.sqrt_alphas_cumprod[prev_timestep], self.sqrt_alphas_cumprod[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        if self.sampler == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.sampler == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.sampler == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                    (sigma_t / sigma_s * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.sampler == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                    (alpha_t / alpha_s) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t
    def multistep_deis_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        One step for the second-order multistep DEIS.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        alpha_t, alpha_s0, alpha_s1 = self.sqrt_alphas_cumprod[t], self.sqrt_alphas_cumprod[s0], self.sqrt_alphas_cumprod[s1]
        sigma_t, sigma_s0, sigma_s1 = self.sigma_t[t], self.sigma_t[s0], self.sigma_t[s1]

        rho_t, rho_s0, rho_s1 = sigma_t / alpha_t, sigma_s0 / alpha_s0, sigma_s1 / alpha_s1

        if self.sampler == "deis":
            def ind_fn(t, b, c):
                # Integrate[(log(t) - log(c)) / (log(b) - log(c)), {t}]
                return t * (-torch.log(c) + torch.log(t) - 1) / (torch.log(b) - torch.log(c))

            coef1 = ind_fn(rho_t, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s0, rho_s1)
            coef2 = ind_fn(rho_t, rho_s1, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s0)

            x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)
            return x_t
        else:
            raise NotImplementedError("only support log-rho multistep deis now")

    def multistep_dpm_solver_second_order_update(
            self,
            model_output_list: List[torch.FloatTensor],
            timestep_list: List[int],
            prev_timestep: int,
            sample: torch.FloatTensor,
            noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        One step for the second-order multistep DPM-Solver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.sqrt_alphas_cumprod[t], self.sqrt_alphas_cumprod[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.sampler == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                        (sigma_t / sigma_s0) * sample
                        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                        (sigma_t / sigma_s0) * sample
                        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                        + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.sampler == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                        (alpha_t / alpha_s0) * sample
                        - (sigma_t * (torch.exp(h) - 1.0)) * D0
                        - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                        (alpha_t / alpha_s0) * sample
                        - (sigma_t * (torch.exp(h) - 1.0)) * D0
                        - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.sampler == "sde-dpmsolver++":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                        (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                        + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                        + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                        + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                        (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                        + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                        + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                        + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.sampler == "sde-dpmsolver":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                        (alpha_t / alpha_s0) * sample
                        - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                        - (sigma_t * (torch.exp(h) - 1.0)) * D1
                        + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                        (alpha_t / alpha_s0) * sample
                        - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                        - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                        + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_deis_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        One step for the third-order multistep DEIS.
        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        alpha_t, alpha_s0, alpha_s1, alpha_s2 = self.sqrt_alphas_cumprod[t], self.sqrt_alphas_cumprod[s0], self.sqrt_alphas_cumprod[s1], self.sqrt_alphas_cumprod[s2]
        sigma_t, sigma_s0, sigma_s1, simga_s2 = self.sigma_t[t], self.sigma_t[s0], self.sigma_t[s1], self.sigma_t[s2]
        rho_t, rho_s0, rho_s1, rho_s2 = (
            sigma_t / alpha_t,
            sigma_s0 / alpha_s0,
            sigma_s1 / alpha_s1,
            simga_s2 / alpha_s2,
        )

        if self.sampler == "deis":

            def ind_fn(t, b, c, d):
                # Integrate[(log(t) - log(c))(log(t) - log(d)) / (log(b) - log(c))(log(b) - log(d)), {t}]
                numerator = t * (
                    torch.log(c) * (torch.log(d) - torch.log(t) + 1)
                    - torch.log(d) * torch.log(t)
                    + torch.log(d)
                    + torch.log(t) ** 2
                    - 2 * torch.log(t)
                    + 2
                )
                denominator = (torch.log(b) - torch.log(c)) * (torch.log(b) - torch.log(d))
                return numerator / denominator

            coef1 = ind_fn(rho_t, rho_s0, rho_s1, rho_s2) - ind_fn(rho_s0, rho_s0, rho_s1, rho_s2)
            coef2 = ind_fn(rho_t, rho_s1, rho_s2, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s2, rho_s0)
            coef3 = ind_fn(rho_t, rho_s2, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s2, rho_s0, rho_s1)

            x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1 + coef3 * m2)

            return x_t
        else:
            raise NotImplementedError("only support log-rho multistep deis now")

    def multistep_dpm_solver_third_order_update(
            self,
            model_output_list: List[torch.FloatTensor],
            timestep_list: List[int],
            prev_timestep: int,
            sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        One step for the third-order multistep DPM-Solver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        )
        alpha_t, alpha_s0 = self.sqrt_alphas_cumprod[t], self.sqrt_alphas_cumprod[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.sampler == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                    - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h ** 2 - 0.5)) * D2
            )
        elif self.sampler == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    - (sigma_t * ((torch.exp(h) - 1.0 - h) / h ** 2 - 0.5)) * D2
            )
        return x_t












if __name__ == '__main__':
    batch_size = 1
    input_height = 180  # 256
    input_width = 180  # 288
    # random from 0-999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.randint(0, 999, (batch_size, 1)).to(device)

    model = Diffuser(
        in_channels=[256, 80],
        num_blocks=[3, 5, 5],
        mid_planes=[64, 128, 256],
        strides=[1, 2, 2],
        upsample_strides=[1, 2, 4],
        num_upsample_filters=[128, 128, 128],
        attention=True
    )
    # model.apply(init_params)
    model.training = False
    cam = torch.randn(batch_size, 256, input_height, input_width, requires_grad=True).to(device)
    lidar = torch.randn(batch_size, 80, input_height, input_width, requires_grad=True).to(device)
    x = [cam, lidar]
    x_pred, diffuse_loss = model(x, t)
