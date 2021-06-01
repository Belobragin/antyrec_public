"""sync pytest for alpm project"""

import time, sys, os, json, cv2, warnings
import numpy as np
from pathlib import Path

from fastapi.testclient import TestClient

warnings.simplefilter("ignore")
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    from alpm.alpmapi import app
    from alpm.algoclass import RecitAlgoName, AlteritAlgoName
    from alpm.lp_util.base_class import BaseAlpm
    from alpm.lp_util.proc import demonstrate_image, from_zip_stream_to_att_data

    #from alpm.recit.discover import ResNet50Discover, TesseractDiscoverLp

    from alpm.changeit.tweaks import CaptchaTweakLp, BasicIterativeAttack,\
                                    FgsmAttack
    from alpm.hardcode import * #module hardcode


base_path = Path(__file__).resolve(strict=True).parent.parent

client = TestClient(app)

#some important tests datastructures:
TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg') 
TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH)
 
def test_changeitpost_basic_iterative_untargeted(): 
    """
    this test runs ~ 300 s.
    """   
    r = {}
    
    for i in range(10):
        print(f'step {i}:')
        ALGO_NAME = AlteritAlgoName.basic_iterative_algo
        ffile =    {'input_image': open(TEST_IMAGE_PATH, 'rb'),
                    "input_image_path": TEST_IMAGE_PATH,
                    "alter_parameters":json.dumps({"acall":False}),}
        r[i] = client.post(f'/alterit/?algo_name={ALGO_NAME}', files = ffile)
        assert r[i].status_code == 200
        i1, i2, i3, i4 = from_zip_stream_to_att_data(r[i])
        assert i1['0'][1] == "wombat"                                        

     
