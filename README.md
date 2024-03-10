# TriNIT Image captioning ml03
## local setup
```
1 - clone the repository in your system
2 - make a virtual environment
3 - run command pip install -r requirements.txt
4 - open develop.py file and run it (in ml_models.py you can change model you are using)
```
## file descriptions
```
## vgg_16_trained.ipynb ---> model trained with base model vgg16 with augmented data

## resnet_50_trained.ipynb -----> model trained with base model resnet 50 on given train data only

## attention_page_caption.ipynb ------> we have tried to write attention mechanism for image captioning but due to some system restrictions not able to trian it

## develop.py -----> streamlit app python file

## embedding_matrix.npy ------> this is matrix for word embeddings which is generated in vgg16_trained.inpynb

## embedding_matrix_s.npy ------> this is matrix for word embeddings which is generated in resnet50_trained.inpynb

## tokenizer.pickle --> tokenizer for vgg16

## tokenizer_s.pickle --> tokenizer for resnet50

## model.tflite ----> trained model with vgg16 as base model

## model_res50.tflite ----> trained model with resnet50 as base model
```

