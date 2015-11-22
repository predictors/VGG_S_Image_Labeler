import numpy as np
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.conv import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
import pickle
import io
import skimage.transform
import matplotlib.pyplot as plt
import json


def save_json_file(**kwargs):


    """
    Saves dictionary in path.
    """

    dict_to_save = kwargs["dict_to_save"]
    path = kwargs["path"]
    with open(path,'wb') as fp:
        json.dump(dict_to_save, fp)

    return


class ImageLabeler:
    
    def __init__(self):
        return

    def predictors_ai_interface(self, **kwargs):

            """
            This is the method used by Predictors.ai to interact with the model.
            Inputs:
            - pipe_id (integer): id of the pipe that has to be used.
            - input_data (dictionary): dictionary that contains the input data. The keys of the dictionary 
            correspond to the names of the inputs specified in models_definition.json for the selected pipe.
            Each key has an associated value. For the input variables the associated value is the value
            of the variable, whereas for the input files the associated value is its filename. 
            - input_files_dir (string): Relative path of the directory where the input files are stored
            (the algorithm has to read the input files from there).
            - output_files_dir (string): Relative path of the directory where the output files must be stored
            (the algorithm must store the output files in there).
            Outputs:
            - output_data (dictionary): dictionary that contains the output data. The keys of the dictionary 
            correspond to the names of the outputs specified in models_definition.json for the selected pipe. 
            Each key has an associated value. For the output variables the associated value is the value
            of the variable, whereas for the output files the associated value is its filename.  
            """

            pipe_id = kwargs['pipe_id']
            input_data = kwargs['input_data']
            input_files_dir = kwargs['input_files_dir']
            output_files_dir = kwargs['output_files_dir']

            output_data = self.predict(pipe_id, input_data, input_files_dir, output_files_dir)

            return output_data
        

    def load_parameters(self):
        
        print("loading parameters...")

        net = {}
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2)
        net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
        net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
        net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
        net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
        net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)
        net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)
        net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
        net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
        net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
        self.output_layer = net['fc8']
        
        self.generate_scores_json()
        self.generate_model_definition()
        
        model = pickle.load(open('vgg_cnn_s.pkl'))
        self.classes = model['synset words']
        self.mean_image = model['mean image']

        lasagne.layers.set_all_param_values(self.output_layer, model['values'])

        return


    def prep_image_from_file(self, filepath):
        ext = filepath.split('.')[-1]
        im = plt.imread(io.BytesIO(open(filepath).read()), ext)
        # Resize so smallest dim = 256, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

        # Central crop to 224x224
        h, w, _ = im.shape
        im = im[h//2-112:h//2+112, w//2-112:w//2+112]

        rawim = np.copy(im).astype('uint8')

        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Convert to BGR
        im = im[::-1, :, :]

        im = im - self.mean_image
        return rawim, floatX(im[np.newaxis])

    
    def generate_scores_json(self):
        
        """
        Calculate scores.
        
        """

        scores = []

        score = {}
        score["name"] = "Top 5 error rate in ILSVRC-2012"
        score["value"] = 13.1
        scores.append(score)
    
        scores_out = {}
        scores_out["scores"] = scores
        scores_out["schema_version"] = "0.02"

        save_json_file(dict_to_save=scores_out, path="./scores.json")
        
        return
    
    
    def generate_model_definition(self):

        """
        Returns model_definition.json dictionary.
        """

        model_definition = {}
        model_definition["name"] = "VGG_S Image Labeler"
        model_definition["schema_version"] = "0.02"
        model_definition["environment_name"] = "python2.7.9_November19th2015"
        model_definition["description"] = "This predictor is based on the example for the python" \
                                          " library Lasagne available at https://github.com/Lasagne/Recipes" \
                                          "/blob/master/examples/ImageNet%20Pretrained%20Network%20(VGG_S).ipynb" \
                                          "<br /><br />" \
                                          "<b>You can find the source code of this onine demo at: " \
                                          "https://github.com/predictors/VGG_S_Image_Labeler</b>" \
                                          "<br /><br />" \
                                          "It demonstrates using a network pretrained on ImageNet for " \
                                          "classification. The model used was converted from the VGG_CNN_S model " \
                                          "(http://arxiv.org/abs/1405.3531) in Caffe's Model Zoo." \
                                          "<br /><br />" \
                                          'For details of the conversion process, see the example notebook "Using' \
                                          'a Caffe Pretrained Network - CIFAR10"'

        model_definition["retraining_allowed"] = False
        model_definition["base_algorithm"] = "Convolutional Neural Network"     
        model_definition["score_minimized"] = ""        

        pipes = self.get_pipes()
        model_definition["pipes"] = pipes

        save_json_file(dict_to_save=model_definition, path="./model_definition.json")

        return


    def get_pipes(self, **kwargs):

        """
        Returns pipes.json dictionary.
        """

        pipes = [ 
                    {
                        "id": 0,
                        "action": "predict",
                        "name":"One by one prediction",
                        "description": "Please upload a png or jpg image.",
                        "inputs": [
                            {
                                "name": "Input image",
                                "type": "file",
                                "extensions": [
                                    "png",
                                    "jpg"
                                ],
                                "required": True
                            }
                        ],
                        "outputs": [
                            {
                                "name": "Predicted labels",
                                "type": "variable",
                                "variable_type": "string"
                            }
                        ]
                    },
                ]

        return pipes

    
    def predict(self, pipe_id, input_data, input_files_dir, output_files_dir):

        image_filepath = input_file_path = input_files_dir + input_data['Input image']
        try:
            rawim, im = self.prep_image_from_file(image_filepath)

            prob = np.array(lasagne.layers.get_output(self.output_layer, im, deterministic=True).eval())
            top5 = np.argsort(prob[0])[-1:-6:-1]

            plt.figure()
            plt.imshow(rawim.astype('uint8'))
            plt.axis('off')
            labels = []
            for n, label in enumerate(top5):
                labels.append("[" + self.classes[label] + "]")
            output = {"Predicted labels": str(labels)[1:-1].replace("'","")}
        except IOError:
            output = {"Predicted labels": "The image you uploaded cannot be read."}
        
        return output
