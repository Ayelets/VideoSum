# VideoSum
Based on https://github.com/ok1zjf/VASNet.
The original code used videos file after preprosesing, Extract feature manipulation and given key points, 
as input. For this reason, the system cannot get and new videos as input.

In our project, we added the preprocessing that was missing on the original network, replace the feature extraction neural network whit inception v3 and retrained. Our system can get a new video and return a summarized video.

Feature extraction inception v3, you can easily replace the feature extractor one another and train 
h5 files https://www.dropbox.com/s/ynl4jsa2mxohs16/data.zip?dl=0 
Summe dataset https://gyglim.github.io/me/vsum/index.html 
h5 files and Summe dataset both are necessary to train the network
Summe dataset contains the original dataset we use it to for feature extraction 
all the other fields are taking from the h5 file of Summe 

A Hebrew article I wrote that explains the algorithm 
https://www.ai-blog.co.il/2019/06/17/ai-%D7%A2%D7%A8%D7%95%D7%9B%D7%99%D7%9D-%D7%9C%D7%96%D7%94-%D7%A2%D7%A8%D7%99%D7%9B%D7%AA-%D7%95%D7%99%D7%93%D7%90%D7%95-%D7%91%D7%A2%D7%96%D7%A8%D7%AA/
