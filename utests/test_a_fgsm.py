"""async pytest for alpm project - fgsm algo """

import time, sys, os, json, cv2, warnings
import numpy as np
from pathlib import Path

from fastapi.testclient import TestClient

import pytest
from httpx import AsyncClient
warnings.simplefilter("ignore")
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    from alpm.alpmapi import app
    from alpm.algoclass import RecitAlgoName, AlteritAlgoName
    from alpm.lp_util.base_class import BaseAlpm
    from alpm.lp_util.proc import demonstrate_image, a_from_zip_stream_to_att_data

    #from alpm.recit.discover import ResNet50Discover, TesseractDiscoverLp

    from alpm.changeit.tweaks import CaptchaTweakLp, BasicIterativeAttack,\
                                    FgsmAttack
    from alpm.hardcode import * #module hardcode


client = TestClient(app)

#some important tets datastructures:
TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg') 
TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH)

async def mygen(u):
    i = -1
    while i<u-1:
        i+=1
        yield i 

@pytest.mark.asyncio
async def test_fgsm():
    """
    async tests alpmapi script (main module) at /alterit/ access point.
    this test runs ~150 s for u = 5 and ~300 s. for u=10 in mygen()
    """ 
    #fgsm algo option:
    r = {}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        
        ALGO_NAME = AlteritAlgoName.fgsm_algo
        ffile =    {'input_image': open(TEST_IMAGE_PATH, 'rb'),
                        "input_image_path": TEST_IMAGE_PATH,
                        "alter_parameters":json.dumps({"acall":True,
                                                    "epsilon":0.01})
                    }
        
        for epsilon_, result_ in zip([0.01, 0.1], ['saluki', 'weimaranner',]):
            r = await ac.post(f'/alterit/?algo_name={ALGO_NAME}', files = ffile)
            assert r.status_code == 200
            i1, i2, i3, i4 = await a_from_zip_stream_to_att_data(r)
            assert i1['0'][1] == result_            
            
        # async for i in mygen(5):
        #     print(f'step {i}:')
        #     #ALGO_NAME = AlteritAlgoName.fgsm_algo
        #     ffile =    {'input_image': open(TEST_IMAGE_PATH, 'rb'),
        #                 "input_image_path": TEST_IMAGE_PATH,
        #                 "alter_parameters":json.dumps({"acall":True,
        #                                             "epsilon":0.01}),}
        #     r[i] = await ac.post(f'/alterit/?algo_name={ALGO_NAME}', files = ffile)
        #     assert r[i].status_code == 200
        #     i1, i2, i3, i4 = await a_from_zip_stream_to_att_data(r[i])
        #     assert i1['0'][1] == "saluki"

    
