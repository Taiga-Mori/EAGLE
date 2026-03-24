from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super().__init__()
        reduced_channels = max(1, int(in_channels * rd_ratio))
        self.reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=1, bias=True)
        self.expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[height, width])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x).view(-1, channels, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        self.se = SqueezeExcitationBlock(out_channels) if use_se else nn.Identity()
        self.reparam_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        self.inference_mode = inference_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.se(self.reparam_conv(x)))


class MobileOne(nn.Module):
    def __init__(
        self,
        num_blocks_per_stage: list[int] | None = None,
        num_classes: int = 90,
        width_multipliers: tuple[float, float, float, float] = (0.75, 1.0, 1.0, 2.0),
        inference_mode: bool = True,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        if num_blocks_per_stage is None:
            num_blocks_per_stage = [2, 8, 10, 1]

        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))

        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            inference_mode=inference_mode,
        )
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0], 0, use_se)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1], 0, use_se)
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            int(num_blocks_per_stage[2] // 2) if use_se else 0,
            use_se,
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_blocks_per_stage[3] if use_se else 0,
            use_se,
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        final_channels = int(512 * width_multipliers[3])
        self.fc_yaw = nn.Linear(final_channels, num_classes)
        self.fc_pitch = nn.Linear(final_channels, num_classes)

    def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int, use_se: bool) -> nn.Sequential:
        strides = [2] + [1] * (num_blocks - 1)
        blocks: list[nn.Module] = []
        for index, stride in enumerate(strides):
            use_se_block = use_se and index >= (num_blocks - num_se_blocks)
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    inference_mode=self.inference_mode,
                    use_se=use_se_block,
                )
            )
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    inference_mode=self.inference_mode,
                    use_se=use_se_block,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc_yaw(x), self.fc_pitch(x)


def mobileone_s0_gaze(num_classes: int = 90) -> MobileOne:
    return MobileOne(num_classes=num_classes, inference_mode=True, width_multipliers=(0.75, 1.0, 1.0, 2.0))
