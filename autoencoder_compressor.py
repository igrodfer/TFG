import string
import numpy as np
import os
from PIL import Image
from ae import AutoEncoder
import torch


class Compressor_Decompressor:
    def __init__(self,model_path,chunk_size=8) -> None:
        self.tile_size = chunk_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.init_autoencoder(model_path)



    def init_autoencoder(self,model_path:string):
        self.model = AutoEncoder.load_autoencoder(model_path)
        self.send_model_to_device()

    def send_model_to_device(self):
        self.model = self.model.to(self.device)

    def send_image_to_device(self,image_tensor:torch.tensor) -> torch.tensor:
        return image_tensor.to(self.device)

    def load_image_as_np(image_path:string)->np.array:
        return np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)

    def store_image_from_np(image_path:string,data:np.array,format='RGB')->Image:
        img = Image.fromarray(data, format)
        img.save(image_path)
        return img

    def segment_image(self,image,pad_type='reflect'):
        img_height,img_width = image.shape[:2]

        # Pads the image so it can be chunked down to a grid even if the size of the image is not
        # divisible by the chunk size
        v_pad = (0,self.tile_size - (img_height % tile_size)) if img_height % tile_size != 0 else (0,0)
        h_pad = (0,tile_size - (img_width % tile_size)) if img_width % tile_size != 0 else (0,0)
            
        image = np.pad(image, (v_pad,h_pad,(0,0)), pad_type)

        img_height , img_width, channels = image.shape

        tiled_array =  image.reshape(img_height // tile_size,
                                    tile_size,
                                    img_width // tile_size,
                                    tile_size,
                                    channels)

        tiled_array = tiled_array.swapaxes(1,2)

        return np.concatenate(tiled_array,axis=0)


    def rebuild_image(self,tile_array,image_size):
        img_height, img_width, channels = image_size
        
        tile_rows = int(np.ceil(img_height/self.tile_size))
        tile_cols = int(np.ceil(img_width/self.tile_size))

        tile_array = tile_array.reshape(tile_rows,
                                        tile_cols,
                                        self.tile_size,
                                        self.tile_size,
                                        channels)
        
        tile_array = np.concatenate(tile_array,axis=1)
        tile_array = np.concatenate(tile_array,axis=1)

        return tile_array[:img_height,:img_width]
    
    def apply_compress_function(self,tile_tensor:torch.tensor) -> torch.tensor:
        return self.model.encode(self.send_image_to_device(tile_tensor))

    def apply_decompression_function(self,encoded_tile_tensor:torch.tensor) -> torch.tensor:
        return self.model.decode(encoded_tile_tensor)

    def make_tensor(self,tile_list_array:np.array)-> torch.tensor:    
        tile_list_array = tile_list_array.swapaxes(2,3)
        tile_list_array = tile_list_array.swapaxes(1,2)
        
        tile_list_tensor = torch.from_numpy(tile_list_array.astype('float32')).reshape(-1,3*self.tile_size*self.tile_size)
        return tile_list_tensor /255


    def retrieve_array(self,decoded_tile_tensor:torch.tensor) -> np.array:
        res = decoded_tile_tensor.detach().cpu().reshape(-1,3,self.tile_size,self.tile_size)
        res = (res.numpy() * 255).astype('uint8')
        res = res.swapaxes(1,2)    
        res = res.swapaxes(2,3)
        return res

