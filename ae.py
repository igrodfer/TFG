from turtle import forward
from numpy import array
import torch.nn as nn
import torch
def load_model(model_type,path,compression_out):
    return model_type.load_model(path,compression_out)

class AutoEncoder(nn.Module):
    def __init__(self, activation=nn.ReLU(), input_size=3*8*8, hidden_sizes=[32*3,8,32*3],compressed_size=None):

        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        if compressed_size != None:
            hidden_sizes[int(len(hidden_sizes)/2)] = compressed_size

        self.init_layers(activation,hidden_sizes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)        
        
    def init_layers(self,activation,hidden_sizes):
        self.activation = activation
        #Encode
        self.fc1 = nn.Linear(self.input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        #Decode
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], self.input_size)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x
    
    def encode(self,x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
    
    def decode(self,x):        
        x = self.activation(x)
        x = self.fc3(x)        
        x = self.activation(x)
        x = self.fc4(x)
        sig = nn.Sigmoid()
        x = sig(x)

        return x
    
    def save_model(self,PATH):
        torch.save(self.state_dict(),PATH)
    
    def load_model(PATH,compression_out):
        model = AutoEncoder(hidden_sizes=[32*3,compression_out,32*3])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model
    
class AutoEncoder_6H(AutoEncoder):
    def __init__(self, activation=nn.ReLU(), input_size=3 * 8 * 8, hidden_sizes=[32*3,24,8,24,32*3],compressed_size=None):
        super().__init__(activation, input_size, hidden_sizes,compressed_size=compressed_size)

    def init_layers(self,activation, hidden_sizes):
        self.activation = activation
        self.fc1 = nn.Linear(self.input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        #Decode
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], self.input_size)
    def encode(self,x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)        
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)

        return x
    
    def decode(self,x):        
        x = self.fc4(x)        
        x = self.activation(x)
        x = self.fc5(x)        
        x = self.activation(x)
        x = self.fc6(x)
        sig = nn.Sigmoid()
        x = sig(x)

        return x
    
    def load_model(PATH,compression_out):
        model = AutoEncoder_6H(hidden_sizes=[32*3,24,compression_out,24,32*3])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model

class AutoEncoder_8H(AutoEncoder):
    def __init__(self, activation=nn.ReLU(), input_size=3 * 8 * 8, hidden_sizes=[32*3,48,24,8,24,48,32*3],compressed_size=None):
        super().__init__(activation, input_size, hidden_sizes,compressed_size=compressed_size)

    def init_layers(self,activation, hidden_sizes):
        self.activation = activation
        #Encode
        self.fc1 = nn.Linear(self.input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        #Decode
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], hidden_sizes[5])
        self.fc7 = nn.Linear(hidden_sizes[5], hidden_sizes[6])
        self.fc8 = nn.Linear(hidden_sizes[6], self.input_size)
    def encode(self,x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)        
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)


        return x
    
    def decode(self,x):        
        x = self.fc5(x)        
        x = self.activation(x)
        x = self.fc6(x)        
        x = self.activation(x)
        x = self.fc7(x)        
        x = self.activation(x)
        x = self.fc8(x)
        x = self.end_step(x)

        return x
    
    def end_step(self,x):
        sig = nn.Sigmoid()
        return sig(x)
    
    def load_model(PATH,compression_out):
        model = AutoEncoder_8H(hidden_sizes=[32*3,48,24,compression_out,24,48,32*3])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model

class AutoEncoder_12H(AutoEncoder):
    def __init__(self, activation=nn.ReLU(), input_size=3 * 8 * 8, hidden_sizes=[160,128,32*3,48,24,8,24,48,32*3,128,160],compressed_size=None):
        super().__init__(activation, input_size, hidden_sizes,compressed_size=compressed_size)

    def init_layers(self,activation, hidden_sizes):
        self.activation = activation
        #Encode
        self.fc1 = nn.Linear(self.input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], hidden_sizes[5])

        #Decode
        
        self.fc7 = nn.Linear(hidden_sizes[5], hidden_sizes[6])
        self.fc8 = nn.Linear(hidden_sizes[6], hidden_sizes[7])
        self.fc9 = nn.Linear(hidden_sizes[7], hidden_sizes[8])
        self.fc10 = nn.Linear(hidden_sizes[8], hidden_sizes[9])
        self.fc11 = nn.Linear(hidden_sizes[9], hidden_sizes[10])
        self.fc12 = nn.Linear(hidden_sizes[10], self.input_size)
    def encode(self,x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)        
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)        
        x = self.activation(x)
        x = self.fc6(x)        
        x = self.activation(x)
        return x
    
    def decode(self,x):        

        x = self.fc7(x)        
        x = self.activation(x)
        x = self.fc8(x)        
        x = self.activation(x)
        x = self.fc9(x)        
        x = self.activation(x)
        x = self.fc10(x)        
        x = self.activation(x)
        x = self.fc11(x)
        x = self.activation(x)
        x = self.fc12(x)
        sig = nn.Sigmoid()
        x = sig(x)

        return x
    
    def load_model(PATH,compression_out):
        model = AutoEncoder_12H(hidden_sizes=[160,128,32*3,48,24,compression_out,24,48,32*3,128,160])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model


class AutoEncoder_8H_noisy(AutoEncoder_8H):
    def __init__(self, activation=nn.ReLU(), input_size=3 * 8 * 8, hidden_sizes=[32*3,48,24,8,24,48,32*3],compressed_size=None,noise_max = 0.0035):
        super().__init__(activation, input_size, hidden_sizes,compressed_size=compressed_size)
        self.noise_factor = noise_max
    def forward(self,x):
        x = self.encode(x)
        x = self.add_noise(x)
        x = self.decode(x)

        return x

    def add_noise(self,x):
        noise = torch.rand_like(x) * self.noise_factor
        return x + noise

class AutoEncoder_8H_Normalnoisy(AutoEncoder_8H_noisy):
    def __init__(self, activation=nn.ReLU(), input_size=3 * 8 * 8, hidden_sizes=[32*3,48,24,8,24,48,32*3],compressed_size=None,noise_mean=0.0175,noise_std=0.011):
        super().__init__(activation, input_size, hidden_sizes,compressed_size=compressed_size, noise_max = None)
        self.noise_mean = noise_mean
        self.noise_std  = noise_std

    def add_noise(self,x):
        noise = torch.randn_like(x) * self.noise_std + self.noise_mean
        return x + noise


class AutoEncoder_8H_nS(AutoEncoder_8H):
    def end_step(self,x):
        return x

    def load_model(PATH,compression_out:int or array[int]):
        model = AutoEncoder_8H_nS(compressed_size=compression_out)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model

class SplitChanelAe(nn.Module):
    def __init__(self,input_size=8*8,compressed_size=[4,2,2],activation=nn.ReLU()):
        self.y_index, self.b_index, self.r_index = 0,1,2

        self.c_sizes = compressed_size
        super(SplitChanelAe, self).__init__()
        self.input_size = input_size
        self.c_sizes = compressed_size
        self.y_encoder = encoder(self.input_size,self.c_sizes[self.y_index])
        self.b_encoder = encoder(self.input_size,self.c_sizes[self.b_index])
        self.r_encoder = encoder(self.input_size,self.c_sizes[self.r_index])

        self.y_decoder = decoder(self.input_size,self.c_sizes[self.y_index])
        self.b_decoder = decoder(self.input_size,self.c_sizes[self.b_index])
        self.r_decoder = decoder(self.input_size,self.c_sizes[self.r_index])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)   

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)

        return x

    def encode(self,x):
        y = x[:,self.y_index,:,:]
        b = x[:,self.b_index,:,:]
        r = x[:,self.r_index,:,:]

        y = self.y_encoder(y)
        b = self.b_encoder(b)
        r = self.r_encoder(r)

        x = torch.cat((y,b,r),1)
        return x
    
    def decode(self,x):
        y_size = self.c_sizes[self.y_index]
        b_size = self.c_sizes[self.b_index]

        y = x[:,0:y_size]
        b = x[:,y_size:(y_size+b_size)]
        r = x[:,(y_size+b_size):]


        y = self.y_decoder(y)
        b = self.b_decoder(b)
        r = self.r_decoder(r)

        return torch.stack((y,b,r),1)

    def load_model(PATH,compression_out:int or array[int]):
        model = SplitChanelAe(compressed_size=compression_out)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model


class encoder(nn.Module):
    def __init__(self,input_size,compressed_size,activation=nn.ReLU,hidden_sizes=[32,24,12]):

        super(encoder, self).__init__()
        self.input_size = input_size
        self.c_size = compressed_size


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)   

        #Encode
        self.fc1 = nn.Linear(self.input_size, hidden_sizes[0])
        self.a1 = activation()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.a2 = activation()
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.a3 = activation()
        self.fc4 = nn.Linear(hidden_sizes[2], self.c_size)
        self.a4 = activation()

    def forward(self,x):
        x = x.view(-1,self.input_size)
        x = self.fc1(x)
        x = self.a1(x)
        x = self.fc2(x)
        x = self.a2(x)
        x = self.fc3(x)
        x = self.a3(x)
        x = self.fc4(x)
        x = self.a4(x)
        return x

class decoder(nn.Module):    
    def __init__(self,output_size,compressed_size,activation=nn.ReLU,hidden_sizes=[12,24,32]):

        super(decoder, self).__init__()
        self.input_size = output_size
        self.c_size = compressed_size


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)   

        #Encode
        self.fc1 = nn.Linear(self.c_size, hidden_sizes[0])
        self.a1 = activation()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.a2 = activation()
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.a3 = activation()
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.a4 = activation()

    def forward(self,x):
        x = self.fc1(x)
        x = self.a1(x)
        x = self.fc2(x)
        x = self.a2(x)
        x = self.fc3(x)
        x = self.a3(x)
        x = self.fc4(x)
        x = self.a4(x)
        return x
