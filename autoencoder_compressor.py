import string
import numpy as np
import os
from PIL import Image
from ae import AutoEncoder
import torch
import ae


class Compressor_Decompressor:
    def __init__(self,model_path:str,model_type,chunk_size=8,compression_out=8,) -> None:
        self.tile_size = chunk_size
        self.compression_out=compression_out
        self.model_type = model_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.init_autoencoder(model_path)

    def init_autoencoder(self,model_path:str):
        model = ae.load_model(self.model_type,model_path,self.compression_out)
        self.model = model.to(self.device)

    def send_image_to_device(self,image_tensor:torch.Tensor) -> torch.Tensor:
        return image_tensor.to(self.device)

    @staticmethod
    def load_image_as_np(image_path:str)->np.ndarray:
        return np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)
        
    @staticmethod
    def store_image_from_np(image_path:str,data:np.ndarray,format='RGB')->Image:
        img = Image.fromarray(data, format)
        img.save(image_path)
        return img

    def segment_image(self,image,pad_type='reflect'):
        img_height,img_width = image.shape[:2]

        # Pads the image so it can be chunked down to a grid even if the size of the image is not
        # divisible by the chunk size
        v_pad = (0,self.tile_size - (img_height % self.tile_size)) if img_height % self.tile_size != 0 else (0,0)
        h_pad = (0,self.tile_size - (img_width % self.tile_size)) if img_width % self.tile_size != 0 else (0,0)
            
        image = np.pad(image, (v_pad,h_pad,(0,0)), pad_type)

        img_height , img_width, channels = image.shape

        tiled_array =  image.reshape(img_height // self.tile_size,
                                    self.tile_size,
                                    img_width // self.tile_size,
                                    self.tile_size,
                                    channels)

        tiled_array = tiled_array.swapaxes(1,2)

        return np.concatenate(tiled_array,axis=0)


    def rebuild_image(self,tile_array:np.ndarray,image_size):
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
    
    def apply_compress_function(self,tile_tensor:torch.Tensor) -> torch.Tensor:
        return self.model.encode(tile_tensor)

    def apply_decompression_function(self,encoded_tile_tensor:torch.Tensor) -> torch.Tensor:
        return self.model.decode(encoded_tile_tensor)

    def make_tensor(self,tile_list_array:np.ndarray)-> torch.Tensor:    
        tile_list_array = tile_list_array.swapaxes(2,3)
        tile_list_array = tile_list_array.swapaxes(1,2)
        
        tile_list_tensor = torch.from_numpy(tile_list_array.astype('float32')).reshape(-1,3*self.tile_size*self.tile_size)
        return tile_list_tensor /255


    def retrieve_array(self,decoded_tile_tensor:torch.Tensor) -> np.ndarray:
        res = decoded_tile_tensor.detach().cpu().reshape(-1,3,self.tile_size,self.tile_size)
        res = (res.numpy() * 255).astype('uint8')
        res = res.swapaxes(1,2)    
        res = res.swapaxes(2,3)
        return res

    @staticmethod
    def scale_array(the_array:np.ndarray):
        max = the_array.max()
        min = the_array.min()
        
        interpolated_array = (the_array - min)* 256 / (max - min) 
        return interpolated_array.to(torch.uint8) , (max,min)

    @staticmethod
    def descale_array(interpolated_array,interval):
        max,min = interval
        the_array = interpolated_array * (max - min) / 256 + min
        return the_array.to(torch.float)


    def compress_image(self,image_path:str,image_np=False):
        if not isinstance(image_np,np.ndarray):
            image_np                = Compressor_Decompressor.load_image_as_np(image_path)
        image_size                  = image_np.shape
        tile_list_np                = self.segment_image(image_np,pad_type='reflect')
        tile_list_tensor            = self.make_tensor(tile_list_np)
        tile_list_tensor_cuda       = self.send_image_to_device(tile_list_tensor)
        compressed_image_tensor     = self.apply_compress_function(tile_list_tensor_cuda)
        clean_c                     = compressed_image_tensor.detach().cpu()
        scaled_array, interval      = Compressor_Decompressor.scale_array(clean_c)
        return scaled_array, interval, image_size #cambio de tipo a float16 para aumentar la compresión perdiendo precision
    
    def decompress_image(self,compressed_image_scaled,interval,destination_path:str,image_size,return_image=False)->np.ndarray:
        if type(compressed_image_scaled) is str: #Si se pasa el tensor como path a archivo se abrirá y copiará
            pass
        compressed_image            = Compressor_Decompressor.descale_array(compressed_image_scaled,interval)
        compressed_image_cuda       = self.send_image_to_device(compressed_image.to(torch.float))
        decompressed_image_tensor   = self.apply_decompression_function(compressed_image_cuda)
        decompresssed_image         = self.retrieve_array(decompressed_image_tensor)
        end_image                   = self.rebuild_image(decompresssed_image,image_size)
        
        if return_image:
            return end_image
        Compressor_Decompressor.store_image_from_np(destination_path, end_image)


