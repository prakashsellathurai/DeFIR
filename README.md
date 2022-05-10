# DeFIR
Content based Image retrieval utilising Nearest Neighbour Search and Deep Learning. 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/prakashsellathurai/defir/main/app.py)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```

## Usage
To view the results from pre-trained model Run the cmd below
```bash
streamlit run app.py
```



## Technical Details:
- Model Architecture:  VGG16 
- Used Dataset : [Fashion Mnist](https://github.com/zalandoresearch/fashion-mnist).
- k-NN indexer:  [Annoy](https://github.com/spotify/annoy) library  by spotify .
- In training stage, the feature vectors for each images in the database are generated from large pretrained models and these vectors's Indices are simultaneously updated on the Nearest neghbour tree Index that are kept in memory.
- In Inference stage, feature vector generated for query image and this feature vector is used as a target in  nearest neighbour search on our NN tree. 



## Advanced Usage
1. To train the model on your own dataset , 
    - go through `train.py`, before running it.
    - `LatentModel` accepts only dataset of  sequence type (eg, numpy arrays and ` tf.data.dataset`)
2. To use custom large pretrained models like transformers or EfficientNet.
    - add them to `get_pretrained_model()` residing in `sim..,retrieval/model.py`
    - make sure to configure the required final layer accordingly. 
> caution: The project is experimental and uses in memory lookup tables and vectors, which may break on bigger datasets with cardinality more than 10K.
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
 [MIT](https://choosealicense.com/licenses/mit/)

## References:
```
Image-based Product Recommendation System with Convolutional Neural Networks
Luyang Chen, Fan Yang, Heqing Yang
CS231n, 2017
(http://cs231n.stanford.edu/reports/2017/pdfs/105.pdf)
```
