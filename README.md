# Wine tasting description generator

The year is 2065 and you're at a glamorous ball with the finest of high society. One of your companions leans over and asks what you think of the wine. Your unsophiticated palate has absolutely no idea -- it tastes like wine. 

Luckily you are prepared and ask your neural implant to generate a wine tasting description. You proceed to share it outloud with your circle of friends, frenemies, acquaintances, and whoever else is standing around. They nod at your wine-tasting wisdom and declare you an expert sommelier! 

(This is a just-for-fun project that aimed to develop an LSTM that generates wine tasting descriptions.)

### Installation

pip install requirements.txt

### How to run

Fire up the wine_tasting_generator.ipynb jupyter notebook.

To train your own model, you will need to source your own GloVe embeddings and training data. Most of the data used for the project is from the [_Wine Reviews_ dataset on Kaggle](https://www.kaggle.com/zynicide/wine-reviews), which was scrapped from the Wine Enthusiast website. I also scrapped data from the [SAQ products website](https://www.saq.com/en/products).
