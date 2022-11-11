import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            # padding, keep the image size constant after next conv2d
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)


#generator

class Generator(nn.Module):

    def __init__(self,in_channels):
        super().__init__()
        #initial convolution
        self.conv=nn.Sequential(nn.ReflectionPad2d(in_channels),
                               nn.Conv2d(in_channels,64,(2*in_channels)+1),
                               nn.InstanceNorm2d(64),
                               nn.ReLU(inplace=True) #final image of same size,channel_size=64
        )
        
        #downsample to 7x7, chanel_size to increase by doble evry time 
        self.down=nn.Sequential(*self.downer(64,128),
                                *self.downer(128,256)
                                
        )
        #does't change the size of the image at all
        self.trans = [ResidualBlock(256) for _ in range(5)]
        self.trans = nn.Sequential(*self.trans)
        
        #upsampling to original image size
        self.up = nn.Sequential(*self.upper(256,128),
                                *self.upper(128,64)
                               
                               )
        
        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64,1, 2*in_channels+1),
            nn.Tanh()
        )
    
   
    
    @staticmethod
    def downer(in_channels,out_channels):
        block=[nn.Conv2d(in_channels,out_channels,3,2,1),
               nn.InstanceNorm2d(out_channels),
               nn.ReLU(inplace=True)]
        return block
    
    @staticmethod    
    def upper(in_channels,out_channels):
        block=[nn.Upsample(scale_factor=2),
               nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
               nn.InstanceNorm2d(out_channels),
               nn.ReLU(inplace=True)
              ]
        return block
    
    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        x = self.trans(x)
        x = self.up(x)
        x = self.out(x)
        return x
        