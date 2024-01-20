
import torch 
import torch.nn as nn
from torchsummary import summary
import numpy as np

class conv3D_block(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(conv3D_block, self).__init__()

        self.conv3D = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv3D(x)
        return x

class up_conv3D_block(nn.Module):

    def __init__(self, in_ch, out_ch, scale_tuple):

        super(up_conv3D_block, self).__init__()

        self.up_conv3D = nn.Sequential(
            nn.Upsample(scale_factor=scale_tuple, mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True), # increasing the depth by adding one below
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up_conv3D(x)
        return x

class SA_UNet_8x8(nn.Module):

    def __init__(self, in_ch_SA=1, out_ch_SA=4):
        super(SA_UNet_8x8, self).__init__()

        filters_3D = [16, 16*2, 16*4, 16*8, 16*16, 16*16] # = [16, 32, 64, 128, 256, 512]
        
        self.Conv3D_1 = conv3D_block(in_ch_SA, filters_3D[0])
        self.Conv3D_2 = conv3D_block(filters_3D[0], filters_3D[1])
        self.Conv3D_3 = conv3D_block(filters_3D[1], filters_3D[2])
        self.Conv3D_4 = conv3D_block(filters_3D[2], filters_3D[3])
        self.Conv3D_5 = conv3D_block(filters_3D[3], filters_3D[4])
        self.Conv3D_6 = conv3D_block(filters_3D[4], filters_3D[5])

        self.MaxPool3D_1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.MaxPool3D_2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.MaxPool3D_3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.MaxPool3D_4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.MaxPool3D_5 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.up_Conv3D_1 = up_conv3D_block(filters_3D[5], filters_3D[4], (1,2,2))
        self.up_Conv3D_2 = up_conv3D_block(filters_3D[4]+filters_3D[4], filters_3D[3], (1,2,2))
        self.up_Conv3D_3 = up_conv3D_block(filters_3D[3]+filters_3D[3], filters_3D[2], (2,2,2))
        self.up_Conv3D_4 = up_conv3D_block(filters_3D[2]+filters_3D[2], filters_3D[1], (2,2,2))
        self.up_Conv3D_5 = up_conv3D_block(filters_3D[1]+filters_3D[1], filters_3D[0], (1,2,2))

        self.Conv3D_final = nn.Conv3d(filters_3D[0]+filters_3D[0], out_ch_SA, kernel_size=1, stride=1, padding=0)
        
    def forward(self, e_SA):

        # SA network's encoder
        e_SA_1 = self.Conv3D_1(e_SA)
        #print("E1:", e_SA_1.shape)
        e_SA = self.MaxPool3D_1(e_SA_1) 
        #print("E2:", e_SA.shape)
        e_SA_2 = self.Conv3D_2(e_SA)
        #print("E3:", e_SA_2.shape) 
        e_SA = self.MaxPool3D_2(e_SA_2)
        #print("E4:", e_SA.shape)
        e_SA_3 = self.Conv3D_3(e_SA)
        #print("E5:", e_SA_3.shape)
        e_SA = self.MaxPool3D_3(e_SA_3)
        #print("E6:", e_SA.shape) 
        e_SA_4 = self.Conv3D_4(e_SA)
        #print("E7:", e_SA_4.shape)
        e_SA = self.MaxPool3D_4(e_SA_4)
        #print("E8:", e_SA.shape) 
        e_SA_5 = self.Conv3D_5(e_SA)
        #print("E9:", e_SA_5.shape) 
        e_SA = self.MaxPool3D_5(e_SA_5)
        #print("E10:", e_SA.shape) 
        e_SA_6 = self.Conv3D_6(e_SA)
        #print("E11:", e_SA_6.shape)

        del(e_SA)

        # SA network's decoder 
        d_SA = self.up_Conv3D_1(e_SA_6)
        #print("D1:", d_SA.shape)
        d_SA = torch.cat([e_SA_5, d_SA], dim=1)
        #print("D2:", d_SA.shape)
        d_SA = self.up_Conv3D_2(d_SA)
        #print("D3:", d_SA.shape)
        d_SA = torch.cat([e_SA_4, d_SA], dim=1)
        #print("D4:", d_SA.shape) 
        d_SA = self.up_Conv3D_3(d_SA)
        #print("D5:", d_SA.shape)
        d_SA = torch.cat([e_SA_3, d_SA], dim=1)
        #print("D6:", d_SA.shape)
        d_SA = self.up_Conv3D_4(d_SA)
        #print("D7:", d_SA.shape)
        d_SA = torch.cat([e_SA_2, d_SA], dim=1)
        #print("D8:", d_SA.shape)
        d_SA = self.up_Conv3D_5(d_SA)
        #print("D9:", d_SA.shape)
        d_SA = torch.cat([e_SA_1, d_SA], dim=1)
        #print("D10:", d_SA.shape) 
        d_SA = self.Conv3D_final(d_SA)
        #print("D11:", d_SA.shape)

        del(e_SA_1, e_SA_2, e_SA_3, e_SA_4, e_SA_5)

        return d_SA


def main():

    model = SA_UNet_8x8()

    print("****** Model Summary ******")

    summary(model, input_size = [(1, 1, 16, 256, 256)], batch_size = -1)

    x = torch.rand(1, 1, 16, 256, 256)
    
    model(x)


if __name__ == '__main__':

    main()

