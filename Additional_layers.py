from typing import Union,Tuple
import torch
import torch.nn as nn

class ConvNorm1d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class ConvTransposeNorm1d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, output_padding: Union[int, Tuple[int]] = 0,
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
    ):
        super().__init__()
        self.dconv = nn.ConvTranspose1d(in_channels,out_channels,kernel_size,stride,padding,output_padding,groups,bias,dilation,padding_mode)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self,x):
        x = self.dconv(x)
        x = self.norm(x)
        return x

class CausalConv1d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.pad = nn.ConstantPad1d(((kernel_size-1)*dilation,0),0.0)
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,padding_mode)

    def forward(self,x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class DilatedCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:Union[int,Tuple[int]],
        num_layers:int = 4,
        dilation_rate:int =2,
        divs:int =4,
        dropout:float=0.1) -> None:
        super().__init__()

        dilations = [dilation_rate**i for i in range(num_layers)]

        _cc = out_channels // divs
        self.init_conv_r = nn.Conv1d(in_channels,out_channels,1)
        self.init_norm_r = nn.BatchNorm1d(out_channels)
        self.init_conv_c = nn.Conv1d(in_channels,_cc,1)
        self.init_norm_c = nn.BatchNorm1d(_cc)
        self.out_conv = nn.Conv1d(in_channels=_cc,out_channels=out_channels,kernel_size=1)
        self.out_norm = nn.BatchNorm1d(out_channels)
        convs = []
        for i in dilations:
            convs.extend([
                CausalConv1d(_cc,_cc,kernel_size,stride=1,dilation=i),
                nn.BatchNorm1d(_cc),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.convs = nn.ModuleList(convs)

    def forward(self,x):
        x_o = self.init_norm_r(self.init_conv_r(x))
        x_c = self.init_norm_c(self.init_conv_c(x))
        for l in self.convs:
            x_c = l(x_c)
        x_relu = torch.relu(x_c)
        x_sigm = torch.sigmoid(x_c)
        x_c = torch.mul(x_relu,x_sigm)
        x_c = self.out_norm(self.out_conv(x_c))
        x = torch.relu(torch.add(x_o,x_c))
        return x

class ChannelAttention(nn.Module): 
    # please refer to SENet.
    def __init__(self,channels:int):
        super().__init__()
        self.convnorm = ConvNorm1d(channels,channels,1)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(channels,channels,1)
    
    def forward(self,x:torch.Tensor):
        x_a = torch.relu(self.convnorm(x))
        x_a = self.GAP(x_a)
        x_a = self.conv(x_a)
        x_a = torch.sigmoid(x_a)
        x = x*x_a
        return x

class DilatedDepthUnit(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        num_layers:int = 4,
        dilation_rate:int =2,
        divs:int =2,
        end_activation=torch.relu,
        channel_attn:bool = True) -> None:
        super().__init__()
        assert (kernel_size%2==1)
        assert (in_channels>1)
        self.ef = end_activation
        
        dilations = [dilation_rate**i for i in range(num_layers)]
        pads = [int((kernel_size*i-i)/2) for i in dilations]
        _c0 = in_channels//divs
        
        
        self.fconv = ConvNorm1d(in_channels,_c0,1)
        self.cconv = ConvNorm1d(in_channels,out_channels,1)

        self.convs = nn.ModuleList([ConvNorm1d(_c0,_c0,kernel_size,dilation=dilations[i],padding=pads[i]) for i in range(num_layers)])

        # channels attention
        self.enabls_ch_attn = channel_attn
        if channel_attn:
            self.ch_attn = ChannelAttention(_c0)

        self.oconv = ConvNorm1d(_c0,out_channels,1)

        self.end_norm = nn.BatchNorm1d(out_channels)

    def forward(self,x):
        x_fork = self.cconv(x)
        
        x_o = self.fconv(x)
        x = torch.relu(x_o)
        for l in self.convs:
            _x = l(x)
            x = torch.relu(torch.add(_x,x_o))
            x_o = _x.clone()
        if self.enabls_ch_attn:
            x = self.ch_attn(x)
        x = self.oconv(x)
        x = self.end_norm(torch.add(x,x_fork))
        x = self.ef(x)
        return x

class DilatedWideUnit(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        num_layers:int = 4,
        dilation_rate:int =2,
        divs:int=2,
        end_activation=torch.relu) -> None:
        super().__init__()
        assert (kernel_size%2==1)
        assert (in_channels>1)
        self.ef = end_activation
        self.num_layers = num_layers
        
        _c0 = in_channels//divs
        dilations = [dilation_rate**i for i in range(num_layers)]
        pads = [int((kernel_size*i-i)/2) for i in dilations]
        self.fconv = ConvNorm1d(in_channels,_c0,1)
        self.cconv = ConvNorm1d(in_channels,out_channels,1)
        self.convs = nn.ModuleList([ConvNorm1d(_c0,_c0,kernel_size,dilation=dilations[i],padding=pads[i]) for i in range(num_layers)])
        self.oconv = ConvNorm1d(_c0,out_channels,1)

        self.end_norm = nn.BatchNorm1d(out_channels)

    def forward(self,x):
        x_fork = self.cconv(x)
        x_init = torch.relu(self.fconv(x))
        x = x_init/self.num_layers
        for l in self.convs:
            x = torch.add(l(x_init)/self.num_layers,x)
        x = torch.relu(x)
        x = self.oconv(x)
        x = torch.add(x_fork,x)
        x = self.end_norm(x)
        x = self.ef(x)
        return x

class ResBlock1d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                channel_divsor:int=2):
        super().__init__()
        assert kernel_size % 2 == 1

        ch = out_channels//channel_divsor
        pad = kernel_size//2
        self.init_conv = ConvNorm1d(in_channels,ch,kernel_size=1)
        self.conv = ConvNorm1d(ch,ch,kernel_size,padding=pad)
        self.out_conv= ConvNorm1d(ch,out_channels,kernel_size=1)
        self.shortcut_conv = self._generate_shortcut(in_channels,out_channels) # skip connection

    def forward(self,x:torch.Tensor):
        h = torch.relu(self.init_conv(x))
        h = torch.relu(self.conv(h))
        h = self.out_conv(h)
        s = self.shortcut_conv(x) 
        y = torch.relu(h+s) # skip connection
        return y

    def _generate_shortcut(self,in_channels: int,out_channels: int):
        if in_channels != out_channels:
            return ConvNorm1d(in_channels,out_channels,kernel_size=1)
        else:
            return lambda x:x
        
class nResBlocks1d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                nlayers: int, channel_divsor:int=2):
        super().__init__()
        assert nlayers >=1
        self.first = ResBlock1d(in_channels,out_channels,kernel_size,channel_divsor)
        layers = [ResBlock1d(out_channels,out_channels,kernel_size,channel_divsor) for _ in range(nlayers-1)]
        self.layers = nn.ModuleList(layers)

    def forward(self,x:torch.Tensor):
        x = self.first(x)
        for l in self.layers:
            x = l(x)
        return x


if __name__ == '__main__':
    from torchsummaryX import summary
    model = nResBlocks1d(8,16,3,2)
    dummy = torch.randn(1,8,16)
    summary(model,dummy)
