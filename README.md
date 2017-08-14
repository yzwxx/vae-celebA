# vae-celebA
Variational auto-encoder trained on celebA . All rights reserved.
## Result:
input image:  
<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/input.png"/>  
</div>  
reconstruction:  
<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/train_49_2914.png"/>  
</div>  
randomly generation:  
<div align="center">
    <img src="https://github.com/yzwxx/vae-celebA/blob/master/train_49_2914_random.png"/>  
</div>  
To run the code, you need to use Tensorflow and Tensorlayer.[how to install Tensorlayer](https://github.com/zsdonghao/tensorlayer)  

## SOME NOTES
This is the code for the paper **[Deep Feature Consistent Variational Autoencoder](https://houxianxu.github.io/assets/project/dfcvae)**  
In loss function we used a vgg loss.Check this [how to load and use a pretrained VGG-16?](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg16.py)if you have trouble reading vgg_loss.py.  

## future work
I'm currently busy with graduation project.An tutorial with more details on this model will be released in two months.
