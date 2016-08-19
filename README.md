#This is a standalone version of OpenCV_contrib Global Patch Collider#

##Overview##
The original code is here
https://github.com/VladX/opencv_contrib

I use OpenCV3.1.0 and VS2015

I have download the pre-trained model from google drive and upload to baiduyun

link：http://pan.baidu.com/s/1i5hmgGd password：stem

*Attention*: the model file has a little problem with my opencv. My opencv can not recognize the yaml file and treat it as xml file because the first line is not correct. If you have the same problem, please revise the first line of the model file like the two images in baidunyun link. (The file is too large, you can use vim to open it. Notepad and Notepad++ can not open it)

##How to use##
*gpc_evaluate.cpp*: GPC.exe < modelpath > < image1path > < image2path > < outputimagepath(optional) > (Because there are too many matched points, I only show about 100 pairs randomly. Press 'N' to show another 100 random pairs.)

*gpc_train.cpp*: I have not tested it yet. The author says the current version need about 4 days to train the model. https://github.com/opencv/opencv_contrib/pull/752#issuecomment-240898100