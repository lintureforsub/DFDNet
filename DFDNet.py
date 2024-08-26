import torch
from torch import nn
from DFDModule import ConvSC
from TemporalModule.mmcls.models.backbones import DFDTemporal

def stride_generator(N, reverse=False):
    strides = [1, 2, 1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class Encoder(nn.Module):
    def __init__(self,
                 C_in,
                 C_hid,
                 N_S,
                 kernel_size):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0], kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2),
            *[ConvSC(C_hid, C_hid, stride=strides[i], kernel_size=kernel_size[i], padding=(kernel_size[i]-1)//2) for i in range(1,N_S)]
        )
        self.relu = nn.ReLU(inplace=True)
        self.hid = C_hid

    def forward(self, x):
        out = []
        out.append(self.enc[0](x))
        for i in range(1, len(self.enc)):
            out.append(self.enc[i](out[i-1]))
        return out


class Decoder(nn.Module):
    def __init__(self,
                 C_hid,
                 C_out,
                 T_in,
                 T_out,
                 N_S,
                 kernel_size):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S)
        self.T_in = T_in
        self.C_hid = C_hid
        self.T_out = T_out
        self.C_out = C_out

        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=strides[i], kernel_size=kernel_size[i], padding=(kernel_size[i] - 1) // 2,
                     transpose=True) for i in range(N_S - 1, 0, -1)],
            ConvSC(C_hid, C_hid, stride=strides[0], kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2,
                   transpose=True)
        )

        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, x):

        data = x[-1]

        for i in range(0, len(self.dec) - 1):

            out = self.dec[i](data)

            data = out + x[-(i + 2)]

        Y = self.dec[-1](data)

        BT, _, H, W = Y.shape

        Y = self.readout(Y)

        return Y


class Mid_Xnet(nn.Module):
    def __init__(self,
                 channel_in,
                 layer,
                 groups=8,
                 layer_config=(2, 8, 2, 8),
                 COF=31/64):
        super(Mid_Xnet, self).__init__()

        self.net = DFDTemporal.DFDT(COF=COF, layer_config=layer_config, in_channels=channel_in,
                                 stem_channels=channel_in, base_channels=channel_in // groups, num_stages=layer,
                                 expansion=groups).to("cuda")

        self.channel_in = channel_in  # T * hid_S

    def forward(self, x):

        for i in range(len(x)):

            BT, _, H, W = x[i].shape

            x[i] = x[i].reshape(-1, self.channel_in, H, W)

        x = self.net(x)

        for i in range(len(x)):

            _, _, H, W = x[i].shape

            x[i] = x[i].reshape(BT, -1, H, W)

        return x

class DFDNet(nn.Module):
    def __init__(self,
                 shape_in,
                 shape_out,
                 COF=31/64,
                 hid_S=64,
                 layer=4,
                 kernel_size=[3, 3, 3, 3],
                 layer_config=(1, 8, 2, 8),
                 groups=8):
        super(DFDNet, self).__init__()

        T, C, H, W = shape_in

        self.T_out, self.C_out, _, _ = shape_out

        self.enc = Encoder(C, hid_S, layer, kernel_size)

        self.hid1 = Mid_Xnet(T * hid_S, layer, groups, layer_config, COF)

        self.dec = Decoder(hid_S, self.C_out, T, self.T_out, layer, kernel_size)

    def forward(self, x_raw):

        B, T, C, H, W = x_raw.shape

        x = x_raw.contiguous().view(B * T, C, H, W)

        data = self.enc(x)

        data = self.hid1(data)

        Y = self.dec(data)

        Y = Y.reshape(B, self.T_out, self.C_out, H, W)

        return Y
