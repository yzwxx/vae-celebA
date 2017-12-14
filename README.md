# vae-celebA
Hereby we present plain VAE and modified VAE model, both of which are trained on celebA dataset to synthesize facial images.
## Result:
### plain VAE
<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/vae_input.png" width="300"/>  
</div>

<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/vae_recon.png" width="300"/>  
</div>  

<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/vae_random.png" width="300"/>  
</div>  

### DFC-VAE
input image:  
<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/input.png" width="300"/>  
</div>  
reconstruction:  
<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/train_49_2914.png" width="300"/>  
</div>  
randomly generation:  
<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/train_49_2914_random.png" width="300"/>  
</div>  

To run the code, you are required to install Tensorflow and Tensorlayer on your machine. **[how to install Tensorlayer](https://github.com/zsdonghao/tensorlayer)**  

## SOME NOTES
This is the code for the paper **[Deep Feature Consistent Variational Autoencoder](https://houxianxu.github.io/assets/project/dfcvae)**  
In loss function we used a vgg loss.Check this **[how to load and use a pretrained VGG-16?](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg16.py)** if you have trouble reading vgg_loss.py.  

## How to Run
Firstly, download the [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [VGG-16 weights](http://www.cs.toronto.edu/%7Efrossard/post/vgg16/).
After installing all the third-party packages required, we can train the models by:  
```python
python train_vae.py # for plain VAE
python train_dfc_vae.py # for DFC-VAE
```
