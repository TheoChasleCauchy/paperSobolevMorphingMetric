# *******************************************
# *        Adapted from cdpam package       *
# *******************************************

import numpy as np
import os
import inspect
import torch
import torch.nn.functional as F
import librosa
import os
import numpy as np
import os
import torch.nn as nn

class base_encoder(nn.Module):
    def __init__(self,dev=torch.device('cpu'),n_layers=20,nefilters=16):
        super(base_encoder, self).__init__()
        self.dev = dev
        nlayers = n_layers
        
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 15
        merge_filter_size = 5
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        
        nchan = nefilters
        for i in range(self.num_layers):
            if i==0:
                chin = 1
            else:
                chin = nchan
            if (i+1)%4==0:
                nchan = nchan*2
            self.encoder.append(nn.Conv1d(chin,nchan,filter_size,padding=filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(nchan))
        
    def forward(self,x):
        input = x
        
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            if (i+1)%4==0:
                x = x[:,:,::2]
        
        x = torch.sum(x,dim=(2))/x.shape[2] # average by channel dimension
        
        dim = 1
        acoustics, content = torch.split(x, x.size(dim) // 2, dim=dim)
        
        return x,acoustics,content
    
class lossnet_dfl(nn.Module):
    def __init__(self,input_size):
        super(lossnet_dfl, self).__init__()
        self.convs = nn.ModuleList()
        self.chan_w = nn.ParameterList()
        for iconv in range(4):
            if iconv==0:
                conv = [nn.Linear(input_size,256),nn.LeakyReLU()]
                self.chan_w.append(nn.Parameter(torch.randn(256),requires_grad=True))
            elif iconv==1:
                conv = [nn.Linear(256,64),nn.LeakyReLU()]
                self.chan_w.append(nn.Parameter(torch.randn(64),requires_grad=True))
            elif iconv==2:
                conv = [nn.Linear(64,16),nn.LeakyReLU()]
                self.chan_w.append(nn.Parameter(torch.randn(16),requires_grad=True))
            elif iconv==3:
                conv = [nn.Linear(16,4)]
                self.chan_w.append(nn.Parameter(torch.randn(4),requires_grad=True))
            self.convs.append(nn.Sequential(*conv))
    
    def forward(self,xref,xper,avg_channel=0):
        # xref and xper are [batch,L]
        dist = 0
        for iconv in range(4):
            xref = self.convs[iconv](xref)
            xper = self.convs[iconv](xper)
            diff = (xper-xref)
            wdiff = diff*self.chan_w[iconv]
            if avg_channel==1:
                wdiff = torch.sum(torch.abs(wdiff),dim=(1))/diff.shape[1] # average by time and channel dimensions
            elif avg_channel==0:
                wdiff = torch.sum(torch.abs(wdiff),dim=(1)) # average by time
            dist = dist+wdiff
        
        return dist

class classifnet(nn.Module):
    def __init__(self,ndim=[16,6],dp=0.1,BN=1,classif_act='no'):
        # lossnet is pair of [batch,L] -> dist [batch]
        # classifnet goes dist [batch] -> pred [batch,2] == evaluate BCE with low-capacity
        super(classifnet, self).__init__()
        n_layers = 2
        MLP = []
        for ilayer in range(n_layers):
            if ilayer==0:
                fin = 1
            else:
                fin = ndim[ilayer-1]
            MLP.append(nn.Linear(fin,ndim[ilayer]))
            if BN==1 and ilayer==0: # only 1st hidden layer
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            elif BN==2: # the two hidden layers
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            MLP.append(nn.LeakyReLU())
            if dp!=0:
                MLP.append(nn.Dropout(p=dp))
        # last linear maps to binary class probabilities ; loss includes LogSoftmax
        MLP.append(nn.Linear(ndim[ilayer],2))
        if classif_act=='sig':
            MLP.append(nn.Sigmoid())
        if classif_act=='tanh':
            MLP.append(nn.Tanh())
        self.MLP = nn.Sequential(*MLP)
        
    def forward(self,dist):
        return self.MLP(dist.unsqueeze(1))

class FINnet(nn.Module):
    def __init__(self,dev=torch.device('cpu'),encoder_layers=12,encoder_filters=24,ndim=[16,6],classif_dp=0.1,classif_BN=0,classif_act='no',input_size=1024,margin=0.1):
        super(FINnet, self).__init__()
        
        self.dev = dev
        self.base_encoder = base_encoder(n_layers=encoder_layers,nefilters=encoder_filters)
        
        self.model_dist = lossnet_dfl(input_size)
        
        self.model_classif = classifnet(ndim=ndim,dp=classif_dp,BN=classif_BN,classif_act=classif_act)
        self.CE = nn.CrossEntropyLoss(reduction='mean')
        self.margin_loss = torch.nn.MarginRankingLoss(margin = margin,reduction='mean')
        
    def forward(self,x1,x2,x3,labels,normalise = 1):
        
        x1_proj,x1_acoustics,x1_content = self.base_encoder.forward(x1.unsqueeze(1))
        
        x2_proj,x2_acoustics,x2_content = self.base_encoder.forward(x2.unsqueeze(1))
        
        x3_proj,x3_acoustics,x3_content = self.base_encoder.forward(x3.unsqueeze(1))
        
        if normalise==1:
            z1 = F.normalize(x1_acoustics, dim=1)
            z2 = F.normalize(x2_acoustics, dim=1)
            z3 = F.normalize(x3_acoustics, dim=1)
        else:
            z1 = x1_acoustics
            z2 = x2_acoustics
            z3 = x3_acoustics
        
        dist_sample1 = self.model_dist.forward(z1,z2)
        dist_sample2 = self.model_dist.forward(z1,z3)
        
        loss = self.margin_loss(dist_sample1,dist_sample2,labels)
        distance = torch.cat((dist_sample1.unsqueeze(-1),dist_sample2.unsqueeze(-1)), 1)
        class_pred = torch.argmin(distance,dim=-1)
        
        return loss,class_pred
        
    
    def grad_check(self,minibatch,optimizer,avg_channel=1):
        xref = minibatch[0].to(self.dev)
        xsample1 = minibatch[1].to(self.dev)
        xsample2 = minibatch[2].to(self.dev)
        labels  = minibatch[3].to(self.dev)
        
        loss,class_pred = self.forward(xref,xsample1,xsample2,labels)
        print('\nbackward on classification loss')
        optimizer.zero_grad()
        loss.backward()
        tot_grad = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                if sum_abs_paramgrad==0:
                    print(name,'sum_abs_paramgrad==0')
                else:
                    tot_grad += sum_abs_paramgrad
            else:
                print(name,'param.grad is None')
        print('tot_grad = ',tot_grad)
        
        norm_type = 2
        loss,class_pred = self.forward(xref,xsample1,xsample2,labels)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            else:
                print(name,'param.grad is None')
        total_norm = total_norm ** (1. / norm_type)
        print('total_norm over all layers ==',total_norm)

class CDPAM():
    def __init__(self, modfolder='models/CDPAM/scratchJNDdefault_best_model.pth', dev='cuda:0'):
        
        self.device = torch.device(dev)
        encoder_layers = 16
        encoder_filters = 64
        input_size = 512
        proj_ndim = [512,256]
        ndim = [16,6]
        classif_BN = 0
        classif_act = 'no'
        proj_dp=0.1
        proj_BN=1
        classif_dp = 0.05
        
        modfolder = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', modfolder))
        #os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth'%(version,net)))
        model = FINnet(dev=self.device,encoder_layers=encoder_layers,encoder_filters=encoder_filters,ndim=ndim, classif_dp=classif_dp,classif_BN=classif_BN,classif_act=classif_act,input_size=input_size)
        state = torch.load(modfolder,map_location="cpu", weights_only=False)['state']
        model.load_state_dict(state)

        model.to(self.device)
        model.eval()
        self.model = model
    
    def forward(self, wav_in=1, wav_out=1):
         
        # input size accepted is [N x Lsize]
        if torch.is_tensor(wav_in) == False:
            audio1 = torch.from_numpy(wav_in).float().to(self.device)
            audio2 = torch.from_numpy(wav_out).float().to(self.device)
        else:
            audio1 = wav_in.float().to(self.device)
            audio2 = wav_out.float().to(self.device)
        
        _,a1,c1 = self.model.base_encoder.forward(audio1.unsqueeze(1))
        a1 = F.normalize(a1, dim=1)
        print('a1:', a1.shape)
        _,a2,c2 = self.model.base_encoder.forward(audio2.unsqueeze(1))
        a2 = F.normalize(a2, dim=1)
        print('a2:', a2.shape)
        dist1 = self.model.model_dist.forward(a1,a2)
        
        return dist1

def load_audio(path):
    
    inputData, fs  = librosa.load(path,sr=22050)
    
    ## convert to 16 bit floating point
    inputData = np.round(inputData.astype(float)*32768)
    
    inputData  = np.reshape(inputData, [-1, 1])
    
    shape_wav = np.shape(inputData)
    
    inputData = np.reshape(inputData, [1,shape_wav[0]])
    
    inputData  = np.float32(inputData)
    
    return inputData