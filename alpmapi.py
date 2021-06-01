"""
FastApi application to process calls from outside
It processes two calls:
1) for image recognition
2) for image alter (attack on ML OR algorythm)
"""
#

import logging
import json, io
import numpy as np
#import re

from fastapi import FastAPI, Query, UploadFile, File, Response #, Form
from fastapi.responses import FileResponse
#from pydantic import BaseModel

import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import alpm.lp_util.base_class

    from alpm.hardcode import *
    from alpm.recit.discover import * 
    from alpm.changeit.tweaks import CaptchaTweakLp, BasicIterativeAttack,\
                            FgsmAttack

    from alpm.algoclass import RecitAlgoName, AlteritAlgoName
    from alpm.lp_util.proc import get_input_par_from_request
    from alpm.lp_util.base_class import AlpmApiCiCd, AlpmError
#from prometheus_fastapi_instrumentator import Instrumentator
#logger = logging.getLogger('alpmlog')

app = FastAPI(
    title = 'alpm package as a service',
    version = VERSION,
    description='API for ML recognition and attacks',
)

@app.get('/')
async def get_main():
    return 'This is a NEW alpmapi application.'

@app.post('/recit/')   #'recognize it '
async def get_recit_algo(input_image: UploadFile = File(...),
                         input_image_path: bytes = File(...),
                         image_parameters: bytes = File(...),
                         algo_name: RecitAlgoName = Query(RecitAlgoName.other_algo, 
                                                          title = 'Recognition algorythm',
                                                          max_length = 100)
                         ):
    """ 
    post for recognition to: .../recit/?algo_name=somevalue 
    
    - algo_name:: function to choose recognition algorythm from .recit
                  it's a query parameter, must be specified
                  by defaul": "'other' 
    """

    input_image, input_image_path, image_parameters = get_input_par_from_request(input_image,
                                                                                input_image_path,
                                                                                image_parameters,
                                                                                )

    if algo_name == RecitAlgoName.other_algo: 
        return {0:'Test option'}
    
    elif algo_name== RecitAlgoName.tessimple_algo:

        return TesseractDiscoverLp(input_image,
                                   input_image_path,
                                   **image_parameters).simple_tesseract()    
    
    elif algo_name == RecitAlgoName.resnet50_algo:
        rec_dict, rec_image = ResNet50Discover(input_image,
                                input_image_path,
                                **image_parameters).resnet50_recognize()
        return rec_dict
    
    else:
        raise AlpmError(UNKNOWN_RECIT_ALGORYTHM_ERROR)
    
    #no predefined recognition algorythm:
    return None

@app.post('/alterit/') #alter it
async def get_altit_algo(input_image: UploadFile = File(...),
                         input_image_path: bytes = File(...),
                         alter_parameters: bytes = File(...),
                         algo_name: AlteritAlgoName = Query(AlteritAlgoName.other_algo, 
                                                          title = 'Alter algorythm',
                                                          max_length = 100)
                         ):
    """ 
    post for alter attack to: .../alterit/?algo_name=somevalue
    I.e.:
    - algo_name:: parameter to choose recognition algorythm from .recit
                  it's a query parameter, must be specified; if not, 'other' by default
    Dictionary parameters in 'alter_parameters':
    - {'target': True} if targeted attack, False (untargeted) by default
    - {'acall': False} if need to perform sync routine; True (async coroutine) by default
                        ! Attn.: sync routine left only for BasicIterative
    """
    input_image, input_image_path, alter_parameters = get_input_par_from_request(input_image,
                                                                                input_image_path,
                                                                                alter_parameters,
                                                                                )
    targeted = alter_parameters.pop("targeted", False)
    async_call = alter_parameters.pop("acall", True)
    
    if algo_name == AlteritAlgoName.other_algo: 
        return {0:'Test option'}
    
    elif algo_name == AlteritAlgoName.captcha_tweak_algo:
        return {0:"Not implemented"} #TODO: program algo

    elif algo_name== AlteritAlgoName.basic_iterative_algo:
        #the only case, when algo can be both sync or async
        if async_call:
            att_streamed_data = await BasicIterativeAttack(input_image,
                                                    input_image_path,
                                                    targeted,                                                     
                                                    **alter_parameters).a_generate_attack()
        else:
            att_streamed_data = BasicIterativeAttack(input_image,
                                                    input_image_path,
                                                    targeted,                                                     
                                                    **alter_parameters).generate_attack()

    elif algo_name== AlteritAlgoName.fgsm_algo:
        att_streamed_data = await FgsmAttack(input_image,
                                            input_image_path,
                                            targeted,                                                     
                                            **alter_parameters).a_generate_attack()
        #TODO: exclude next line after algorythm implementation:
        #return att_streamed_data
    
    else:
        raise AlpmError(UNKNOWN_ALTER_ALGORYTHM_ERROR)
    
                        
    return Response(att_streamed_data.getvalue(), 
                        media_type="application/x-zip-compressed", 
                        headers={'Content-Disposition': f'attachment;filename={SEND_BACK_TEMP_ATT_FILENAME}'
                        })

if __name__ == '__main__':
    alpm_api_ci_cd = AlpmApiCiCd(debug=True)
    alpm_api_ci_cd.run()
##

