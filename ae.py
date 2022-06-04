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
    
class Autoencoder_5hidden(AutoEncoder):
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

        return x
    
    def decode(self,x):        
        x = self.activation(x)
        x = self.fc4(x)        
        x = self.activation(x)
        x = self.fc5(x)        
        x = self.activation(x)
        x = self.fc6(x)
        sig = nn.Sigmoid()
        x = sig(x)

        return x
    
    def load_model(PATH,compression_out):
        model = Autoencoder_5hidden(hidden_sizes=[32*3,24,compression_out,24,32*3])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model

class Autoencoder_7hidden(AutoEncoder):
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
        model = Autoencoder_7hidden(hidden_sizes=[32*3,48,24,compression_out,24,48,32*3])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model

class AutoEncoder_11H(AutoEncoder):
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
        model = AutoEncoder_11H(hidden_sizes=[160,128,32*3,48,24,compression_out,24,48,32*3,128,160])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model


class AutoEncoder_7H_noisy(Autoencoder_7hidden):
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

class AutoEncoder_7H_Normalnoisy(AutoEncoder_7H_noisy):
    def __init__(self, activation=nn.ReLU(), input_size=3 * 8 * 8, hidden_sizes=[32*3,48,24,8,24,48,32*3],compressed_size=None,noise_mean=0.0175,noise_std=0.011):
        super().__init__(activation, input_size, hidden_sizes,compressed_size=compressed_size, noise_max = None)
        self.noise_mean = noise_mean
        self.noise_std  = noise_std

    def add_noise(self,x):
        noise = torch.randn_like(x) * self.noise_std + self.noise_mean
        return x + noise


class AutoEncoder_7H_nS(Autoencoder_7hidden):
    def end_step(self,x):
        return x