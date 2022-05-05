import torch.nn as nn
import torch
def load_model(model_type,path,compression_out):
    return model_type.load_autoencoder(path,compression_out)

class AutoEncoder(nn.Module):
    def __init__(self, activation=nn.ReLU(), input_size=3*8*8, hidden_sizes=[32*3,8,32*3]):

        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.init_layers(activation,hidden_sizes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)        
        
    def init_layers(self,activation,hidden_sizes):
        self.activation = activation
        self.fc1 = nn.Linear(self.input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
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
    
    def load_autoencoder(PATH,compression_out):
        model = AutoEncoder(hidden_sizes=[32*3,compression_out,32*3])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model
    
class Autoencoder_5hidden(AutoEncoder):
    def __init__(self, activation=nn.ReLU(), input_size=3 * 8 * 8, hidden_sizes=[32*3,24,8,24,32*3]):
        super().__init__(activation, input_size, hidden_sizes)

    def init_layers(self,activation, hidden_sizes):
        self.activation = activation
        self.fc1 = nn.Linear(self.input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
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
    
    def load_autoencoder(PATH,compression_out):
        model = Autoencoder_5hidden(hidden_sizes=[32*3,24,compression_out,24,32*3])
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model