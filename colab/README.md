# Colab Notebook
[![Open In Colab]((https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U7OR6zdhiCzfeQ-x77Sl0suE9ijRknnv#scrollTo=y0L_bNjddd2y)

This is the folder for the colab version of this repository. Some minor changes had to be
made to main.py and models.py. When you load into the colab 
notebook and clone this repository you will need to go into
main.py by double clicking it and replace 
```python
sys.path.insert(1, './code')
```
with this
```python
sys.path.insert(1, './Pytorch-TecoGAN/code/')
```
If you are using a different size crop which I recommend 
because colab has more gpu memory you will have to change 
the input size of the discriminator linear layer. Double click
models.py and go to line 123. For 256x256 high resolution images you need to change the input dim to 192 like this:
```python
        self.fc = denselayer(192, 1)

```
This takes about 15gb of data and 2 minutes and 40 seconds per epoch on a Tesla T4

#Kaggle Dataset
You will also need to download the kaggle dataset I created 
[here](https://www.kaggle.com/gtownfoster/ucf101-images-for-tecogan-pytorch).
The command line code to do it is already in the colab but you will 
need to download the kaggle.json file from your kaggle profile and upload it to colab.
You can find out more about this [here](https://www.kaggle.com/docs/api). Once you upload
it to google colab the code should do the rest. 

