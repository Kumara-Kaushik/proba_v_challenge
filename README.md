# Proba v challenge:
The code in this repo documents my approach at attempting to solve the proba-v vhallenge. 
Note: This is an experimental approach using Unet Architecture to explore a new way to solve the challenge. The winners of the challenge released their codes here and here, These models are end to end networks specifically trained for the challenge. Instead I will be using a resnet pre-trained Unet model approach to see if we can obtain similar or better results with a more generalized architecture. 

## Problem statement:
We are given multiple images of each of 78 Earth locations and we are asked to develop an algorithm to fuse them together into a single one. The result should be a "super-resolved" image that is checked against a high resolution image taken from the same satellite, PROBA-V. The 'V' stands for Vegetation, which is the main focus of the on-board instruments.

![satellite image](/pictures/proba_2.png)

## Data preparation:
We are provided with a varying number of low resolution images (16 bit gray scale images) for each scene. They also come with corresponding masks representing which areas of the image are covered with clouds and other similar obstructions. For example:
![mask display](/pictures/display_1.png)
![mask display1](/pictures/display_3.png)

As shown in the figures below, The way we process the data is as follows:
1. For each scene, We read all the given low-resolution images, and their corresponding maps and store then in numpy array format. 
![1st image](/pictures/draw_1.png)

2. We also create an empty numpy array to represent the final training image.
3. Looping through each pixel index, We will compare the low resolution images with their corresponding masks, take the pixel values at that index of only the images whose masks shows us they aren't obstructed.
4. We then take the median of these pixels and replace them in the corresponding index of the final training image. 
![2nd image](/pictures/draw_2.png)

We do this because taking the median or average across all the images will include pixel values of all the obstructed parts of the image as well. By taking the average of only the unobstructed pixels, we get a better, more accurate representation of the original scene.

The below figure shows a post processed training LR image which will be used to train the model to predict the corrosponding HR image.
![compare display](/pictures/display_2.png)

We save all the modified low-resolution images into a folder called LR_imgs with each image's name corresponding to its scene name. We similarly move all the HR images into a folder called HR_imgs with their names changed to match their corresponding scenes.

## Model Architecture:
The model architecture I chose to use here is a Unet based architecture.

As shown below, A Unet based architecture basically consists of two parts. A down sampler nerwork and an upsampler network. But the most important feature which makes this a really good choice for super resulution is its skip connection between layers. This allows the model to retain information from all stages of the pipeline which makes it a really good choice for super resolution tasks. we will be using a resnet-34 architecture pre-trained on the Imagenet database as the down sampler and upsampler of the Unet design. The pre-trained weights should help us learn the image features a lot better than than an non pre-trained model.
![img_3](/pictures/draw_3.png)

But one of the drawbacks of the pre-trained model is that the Imagenet database on which it was trained on, only has images of more general things like cats, dogs, humans, etc. How can this help in salatille imagery?

Well, it isnt entirely bad news. Infact, its not bad news at all! Even though the model was pretrained on more general images, The pre trained weights contain valuable information on how to choose features at the low end. We use this to our advantage. We will train on top of these weights to get faster gradient decent and modify the last few layers to correctly predict satellite images on a higher level. 

We will initially train the model with low resolution images and to output images with the same resolution and gradually increase the output resolution to match the size of the ground truth high resolution images. We do this, so that the model can gradually learn the features and get better at predicting bigger size images with more accuracy.

## Loss Function:

While The compitition asks us to use a specific type of loss function called cPSNR, I chose to rather use a more detailed loss function called Perceptual Loss function derived from this paper. The reason being, while the cPSNR seems to be a good loss metric for the given data, but PSNR can vary wily between two similar images. Sometimes, we get a good PSNR value between two images which have some obvious diference between them. Therefore, using a more sophisticated different loss function which will enable us to effectively train a model which can more accuratly judge if the image its predicting has the right features. Not only that, We will making making slight modifications to the loss by also adding gram matrix style loss to help us accuratly predict images with the right style. The loss function we will be using in this paper will be further explained below.

![loss_function](/pictures/proba_1.png)

As shown, in the above figure, The loss was originally implemented on a VGG network, the weights at all the loss calculating layers remain the same, therefore, we will be taking all the relavent layers and build only the [arts necessary to create a loss function which best replicates the above loss function. We will also be adding an additional loss function called gram matrix loss. Gram matrix loss is generally employed in style transfer algorithms. We can use this to our advantage to effectively transfer the right image style when generating a super resolution image.


## Results:
The results are quite interesting to be honest. The images below show the super resolution predicted image, the HR image and the corresponding bileaner upsampled image. Clearly The model produces more rich and accurate representation of the the ground truth image compared to the bilinear upsampled image. Yet, the model predicted test set returns a worst score than the base score when using the compititon's suggested loss function. 

![output_1](/pictures/plot_1.png)
![output_2](/pictures/plot_2.png)

The below table shows the scores of the predicted test set with the scores of bilinear upsampled test set. 

![table_1](/pictures/table_1.png)

While we could say the loss function the compitition uses does not accuratly determine the right solution, There are other factors we need to consider improving which could help us increase the score. for instance:
1. The difference in brightness seems to play a major role in the the calculation of the the cPSNR scores. 
2. We do not make use of the masks provided for the HR images. 
3. We do not account for the minor shifts which are present between the LR and HR images.

All this  could be a major factor which will lead to score improvement. Yet, I can safely say that the model produces clearer high definition images than a bilinear upsampled images. 


## Conclusion:
This was an experimental implementation to find a differnt solution to the problem. As mentioned above, It will definitly need more modifications and experimentations. While I will continue to work on it while time allows me, It was quite interesting to see such good results without the need for complex ground up AI architectures and minimal data preprocessing. 

The challenge's winning solutions were built with special ground up architectures which take care of the points we missed including in this experimentation, While I experimented with the Unet Architecture here, I would rather implement their solution for an immediate real world project. 

## Repo Instructions:

This Implementation requires you to have the fastai library installed. Installing the dependencies in the requirments.txt file should be enough to run the repo. It is also recommended to install the dependencies in a virtual environment.

### Data Preparation:
* Once the repo has been cloned, enter the base directory and run the f0llowing command to download and prepare the dataset. (set download to False if you already have the data unzipped in the probav_data folder.)
  
  python generate_data.py --download=True
  
### Training the model:
* In order to train the model, You can run the Jupyter notebook called "train_probav_model.ipynb" from top to bottom in order. Or you can skip this step and use a saved model I have trained using the same method by downloading the export model file from [here](https://drive.google.com/file/d/1KFIL-GI4FYwrZNBOaeFLO2Qdv1zr73yv/view?usp=sharing) called "export.pkl", placing it in the "model_data" folder and running the inference script.

### Generating submission file and running inference on single image:
* In order to generate the submission file, Please download the trained model export file from [here](https://drive.google.com/file/d/1KFIL-GI4FYwrZNBOaeFLO2Qdv1zr73yv/view?usp=sharing) and place it in the "model_data" folder.

* In order to genereate the super resolution images for all the images in the test folder and zip it into a submission file, run the following command:
  
  python predict.py save_final_images
  
* In order to do predictions on single images and compare it with its bilinear upsampled counterpart and HR ground truth, You can run the following command and specifying the image path.
  
  python predict_one.py --path=<image_root_path>
  
## References

1. https://arxiv.org/abs/1603.08155
2. https://github.com/ElementAI/HighRes-net/blob/master/README.md
3. https://github.com/diegovalsesia/deepsum
4. https://towardsdatascience.com/enhancing-satellite-imagery-with-deep-multi-temporal-super-resolution-24f08586ada0
5. https://kelvins.esa.int/proba-v-super-resolution/data/
6. https://github.com/cedricoeldorf/proba-v-super-resolution-challenge
7. https://towardsdatascience.com/loss-functions-based-on-feature-activation-and-style-loss-2f0b72fd32a9
8. https://arxiv.org/abs/1508.06576
9. https://docs.fast.ai/


