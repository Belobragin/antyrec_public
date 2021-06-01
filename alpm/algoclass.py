""" classes define algorythms for recognition and alter """


from enum import Enum

class RecitAlgoName(str, Enum):
    """ algorythms to recognize image (defence) """
    other_algo = 'other' #Algorythm in developing
    tessimple_algo = 'tessimple' #Tesseract - no preprocessing
    resnet50_algo = 'NN_ResNet50' #ResNet50 to recognize ImageNet

class AlteritAlgoName(str, Enum):
    """algorythms to change image (attack) """
    other_algo = 'other' #Algorythm in developing
    captcha_tweak_algo = 'captcha_tw' # CaptchaTweakLp attack
    basic_iterative_algo = 'bim' # Basic iterative algorythm, both targeted and untargeted
    fgsm_algo = 'fgsm' #fast gradient sign method attack