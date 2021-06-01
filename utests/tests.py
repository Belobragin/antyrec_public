"""unittests for alpm project"""

import unittest, time, sys, os, requests, json, cv2, warnings
from pathlib import Path

from fastapi.testclient import TestClient
from httpx import AsyncClient

from alpmapi import app
from alpm.algoclass import RecitAlgoName, AlteritAlgoName
from alpm.lp_util.base_class import BaseAlpm
from alpm.lp_util.proc import demonstrate_image, from_zip_stream_to_att_data,\
                              from_zip_stream_to_att_data

from alpm.recit.discover import ResNet50Discover, TesseractDiscoverLp

from alpm.changeit.tweaks import CaptchaTweakLp, BasicIterativeAttack,\
                                 FgsmAttack

warnings.simplefilter("ignore")

#from selenium import webdriver
#from selenium.webdriver.common.keys import Keys

from alpm.hardcode import * #module hardcode
#from .hardcode import * #tests hardcode
#from webdriver_manager.chrome import ChromeDriverManager
#print(os.getcwd())

base_path = Path(__file__).resolve(strict=True).parent.parent

class AlpmModuleGeneralTest(unittest.TestCase):
    """
    tests:
    1. general modules import
    2. all error class validness
    3. general class validness
    4. general import validness
    """      
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_error_valid_exception(self):
        """ tests validness of AlpmError"""
        try:
            from alpm.lp_util.base_class import AlpmError
        except:
            self.fail('Can not import  AlpmError class')
        with self.assertRaises(AlpmError):
            raise(AlpmError('Test error'))
           
    def test_modules_import(self):
        """tests modules import"""
        try:
            from alpm import hardcode
        except Exception as ee:
            self.fail('alpm.hardcode import failed')
        try:
            from alpm import algoclass
        except Exception as ee:
            self.fail('alpm.algoclass import failed')
        try:
            #from alpm import alpmapi - structure changed for CI/CD since 27.03.2021
            import alpmapi
        except Exception as ee:
            self.fail('alpmapi import failed')
        try:
            from alpm.recit import discover
        except Exception as ee:
            self.fail('discover import failed')
        try:
            from alpm.changeit import tweaks
        except Exception as ee:
            self.fail('alpm.changeit.tweaks module import failed')
        try:
            from alpm.changeit.tweaks import CaptchaTweakLp, BasicIterativeAttack
        except Exception as ee:
            self.fail('alpm.changeit.tweaks class import failed')
        try:
            from alpm.recit.discover import ResNet50Discover, TesseractDiscoverLp
        except Exception as ee:
            self.fail('alpm.recit.discover class import failed') 
    
    def test_general_module_classes(self):
        """tests classes of alpm/recit.discover"""
        from alpm.lp_util.base_class import AlpmError
        with self.assertRaises(AlpmError):
            temp_inst = BaseAlpm()        
        #test instances:
        #base class BaseAlpm
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', '12347.jpeg')
        temp_inst = BaseAlpm(input_image_path = TEST_IMAGE_PATH)        
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        self.assertTrue(temp_inst)
        TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH)        
        temp_inst = BaseAlpm(input_image = TEST_IMAGE)        
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        self.assertTrue(temp_inst) 


class AlpmModuleRecitTests(unittest.TestCase):
    """
    tests:
    1. recit modules import
    2. recit class validness
    3. recit import validness
    4. recit algorythms correctness
    """      
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_discover_module_classes(self):
        """tests classes of alpm/recit.discover"""
        from alpm.lp_util.base_class import AlpmError
        with self.assertRaises(AlpmError):
            temp_inst = TesseractDiscoverLp()
        with self.assertRaises(AlpmError):
            temp_inst = ResNet50Discover() 
        
        #test instances:
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', '12347.jpeg')        
        TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH) 
        #Tesseract class:        
        temp_inst = TesseractDiscoverLp(lp_image = TEST_IMAGE)      
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        temp = temp_inst.simple_tesseract()
        self.assertTrue(temp)
        #ResNet50 class:
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg')        
        temp_inst = ResNet50Discover(input_image_path = TEST_IMAGE_PATH)        
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        temp = temp_inst.resnet50_recognize()
        self.assertTrue(temp)
        self.assertTrue(len(temp) ==2)

    def test_discover_module_methods_correctness(self):
        """ tests algorythms correctness in discover module classes methods """

        #test TesseractDiscoverLp.simple_tesseract()
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', '12347.jpeg')
        temp_inst = TesseractDiscoverLp(lp_image_path = TEST_IMAGE_PATH) #, param = 7)        
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        temp = temp_inst.simple_tesseract()
        self.assertIn('E210PE', temp['0'])
        
        #ResNet50Discover.resnet50_recognize()
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg')               
        temp_inst = ResNet50Discover(input_image_path = TEST_IMAGE_PATH)        
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        temp = temp_inst.resnet50_recognize()
        self.assertTrue(temp[0][0][:3] == ('n02395406', 'hog', 341))


class AlpmChangeitGeneralTests(unittest.TestCase):
    """
    tests:
    1. changeit class validness
    """      
    def setUp(self):
        warnings.simplefilter("ignore")
    
    def test_tweaks_module_classes(self):
        """tests classes of alpm/changeit.tweaks"""
        
        from alpm.lp_util.base_class import AlpmError

        with self.assertRaises(AlpmError):
            temp_inst = CaptchaTweakLp()
        with self.assertRaises(AlpmError):
            temp_inst = BasicIterativeAttack()
        with self.assertRaises(AlpmError):
            temp_inst = FgsmAttack()   

        #test instances:
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', '12347.jpeg')        
        TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH) 

        #CaptchaTweakLp class:        
        temp_inst = CaptchaTweakLp(lp_image = TEST_IMAGE)      
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        #temp = temp_inst.simple_tesseract()
        #self.assertTrue(temp)

    def test_tweaks_module_class_BasicIterativeAttack_default(self):
        """SimpleImagenetAttack class instance test with default parameters"""
        
        from tensorflow.keras.applications import ResNet50

        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg')
        TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH)         
        #temp_inst = BasicIterativeAttack(input_image_path = TEST_IMAGE_PATH)       
        temp_inst = BasicIterativeAttack(input_image = TEST_IMAGE)
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        self.assertTrue(isinstance(temp_inst.output, type(None)))
        self.assertFalse(temp_inst.target_goal)
        self.assertTrue(temp_inst.learning_rate == BASE_LEARNING_RATE)
        self.assertTrue(temp_inst.eps == BASE_EPS)
        self.assertTrue(temp_inst.steps == 50)
        self.assertTrue(temp_inst.model.name == 'resnet50')
        self.assertTrue(temp_inst.class_idx == 341)


    def test_tweaks_module_class_BasicIterativeAttack_nondefault(self):
        """SimpleImagenetAttack class instance test with non-default parameters"""

        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg')
        TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH)         
        temp_inst = BasicIterativeAttack(input_image_path = TEST_IMAGE_PATH,
                                        target_goal = True,
                                        learning_rate = 0.2,
                                        eps = 0.15,
                                        steps = 500,
                                        class_idx = 111, #Att:! this is a kwarg value, not constructor
                                        )      
        self.assertFalse(isinstance(temp_inst.image, type(None)))
        self.assertTrue(temp_inst.target_goal)
        self.assertTrue(temp_inst.learning_rate == 0.2)
        self.assertTrue(temp_inst.eps == 0.15)
        self.assertTrue(temp_inst.class_idx == 111)
        self.assertTrue(temp_inst.steps == 500)


class AlpmScriptRecitTests(unittest.TestCase):
    """
    tests alpmapi script (main module) at /recit/ access point:
    1. app fastapi validness
    2. algorythms correctness
    """      
    
    def setUp(self):
        self.client = TestClient(app)
        warnings.simplefilter("ignore")
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg') 
        TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH)
        self.ffile =    {'input_image': open(TEST_IMAGE_PATH, 'rb'),
                        "input_image_path":TEST_IMAGE_PATH,
                        "image_parameters": '{}',
                        }
        #print(vars(self.client))
        
    #def tearDown(self):

    def test_read_main(self):
        response = self.client.get("/")
        time.sleep(10)
        self.assertTrue(response.status_code, 200)
        self.assertIn(response.json(), "This is an alpmapi application.")
    
    def test_recitpost_other(self):       
        #other option:
        ALGO_NAME = RecitAlgoName.other_algo
        r = self.client.post(f'/recit/?algo_name={ALGO_NAME}', files = self.ffile)
        self.assertTrue(r.status_code, 200)
        temp = r.json()
        self.assertFalse(isinstance(temp, type(None)))
        self.assertEqual(temp['0'], 'Test option')

    def test_recitpost_simpletesseract(self):    
        #tessimple_algo option:
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', '12347.jpeg')
        self.ffile['input_image+path'] = TEST_IMAGE_PATH
        self.ffile['input_image'] = open(TEST_IMAGE_PATH, 'rb')
        ALGO_NAME = RecitAlgoName.tessimple_algo
        r = self.client.post(f'/recit/?algo_name={ALGO_NAME}', files = self.ffile)
        self.assertTrue(r.status_code, 200)
        temp = r.json()
        self.assertIn('E210PE', temp['0'])

    def test_recitpost_resnet(self):    
        #resnet50_algo option:
        ALGO_NAME = RecitAlgoName.resnet50_algo
        r = self.client.post(f'/recit/?algo_name={ALGO_NAME}', files = self.ffile)
        self.assertTrue(r.status_code, 200)
        temp = r.json()
        self.assertTrue(len(temp) == 3)
        self.assertTrue(temp['0'][:3] == ['n02395406', 'hog', 341])


class AlpmChangeitAlgorythmTests(unittest.TestCase):
    """
    test changeit module algorythms correctness
    """      
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_tweaks_module_simple_captcha_method(self):
        #test CaptchaTweakLp.simple_captcha()
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', '12347.jpeg')
        temp_inst = CaptchaTweakLp(lp_image_path = TEST_IMAGE_PATH)
        temp = temp_inst.simple_captcha()
        #self.fail('No such method: simple_captcha - yet')

    def test_tweaks_module_untargeted_attack_method(self):    
        #SimpleImagenetAttack.untargeted_attack()
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg')               
        temp_inst = BasicIterativeAttack(input_image_path = TEST_IMAGE_PATH)        
        temp = from_zip_stream_to_att_data(temp_inst.generate_attack())
        self.assertTrue(temp[0]['0'][1] == "wombat")

    def test_tweaks_module_targeted_attack_method(self):    
        #SimpleImagenetAttack.targeted_attack()
        # !Attn.: long-running test
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg')               
        temp_inst = BasicIterativeAttack(input_image_path = TEST_IMAGE_PATH,
                                         #learning_rate =  0.1,
                                         targeted = True,
                                         target_goal = 189,
                                         verbose_step = 50,
                                         steps = 200)        
        temp = from_zip_stream_to_att_data(temp_inst.generate_attack())
        #print(temp[0])
        #demonstrate_image('new window', temp[2])
        #demonstrate_image('delta window', temp[3])
        self.assertTrue(temp[0]['0'][1] == "Lakeland_terrier")


class AlpmChangeitScriptTestsButImageNet(unittest.TestCase):
    """
    tests alpmapi script (main module) at /alterit/ access point.
    this class DOES NOT test ImageNet attacks
    """      
    
    def setUp(self):
        self.client = TestClient(app)
        warnings.simplefilter("ignore")
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg') 
        TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH)
        self.ffile =    {'input_image': open(TEST_IMAGE_PATH, 'rb'),
                        "input_image_path": TEST_IMAGE_PATH,
                        "alter_parameters":'{}'}
    
    def test_changeitpost_other(self):       
        #other option:
        ALGO_NAME = AlteritAlgoName.other_algo
        r = self.client.post(f'/alterit/?algo_name={ALGO_NAME}', files = self.ffile)
        self.assertTrue(r.status_code, 200)
        temp = r.json()
        self.assertFalse(isinstance(temp, type(None)))
        self.assertEqual(temp['0'], 'Test option')      
    
    def test_changeitpost_captcha_tweak(self):    
        #captcha_tweak algo option:
        #TODO: rewrite test after algo implementation
        ALGO_NAME = AlteritAlgoName.captcha_tweak_algo
        r = self.client.post(f'/alterit/?algo_name={ALGO_NAME}', files = self.ffile)
        self.assertTrue(r.status_code, 200)
        temp = r.json()
        self.assertFalse(isinstance(temp, type(None)))
        self.assertEqual(temp['0'], 'Not implemented')  
    

class AlpmChangeitScriptImageNetTests(unittest.TestCase):
    """
    tests alpmapi script (main module) at /alterit/ access point.
    
    !ATTN: tests with new output format for alpmapi response - do work after 16.04.2021
    """      
    
    def setUp(self):
        self.client = TestClient(app)
        warnings.simplefilter("ignore")
        TEST_IMAGE_PATH = os.path.join('alpm/test_data', 'pig.jpg') 
        TEST_IMAGE = cv2.imread(TEST_IMAGE_PATH)
        self.ffile =    {'input_image': open(TEST_IMAGE_PATH, 'rb'),
                        "input_image_path": TEST_IMAGE_PATH,
                        "alter_parameters":'{}'}
    
    def test_changeitpost_basic_iterative_archive(self):
        """ test response with files archive """
        import requests
        import numpy as np
        target_path = 'alaska.zip'
        ALGO_NAME = AlteritAlgoName.basic_iterative_algo
        r = self.client.post(f'/alterit/?algo_name={ALGO_NAME}', files = self.ffile)
        self.assertTrue(r.status_code, 200)
        i1, i2, i3, i4 = from_zip_stream_to_att_data(r,
                                                                            target_path = 'arctic.zip',
                                                                            )
        
        #demonstrate_image('new window', i3)
    
    def test_changeitpost_basic_iterative_untargeted(self):    
        #captcha_tweak algo option:

        ALGO_NAME = AlteritAlgoName.basic_iterative_algo
        r = self.client.post(f'/alterit/?algo_name={ALGO_NAME}', files = self.ffile)
        self.assertTrue(r.status_code, 200)
        i1, i2, i3, i4 = from_zip_stream_to_att_data(r,
                                                    )
        #print(i1)
        self.assertTrue(i1['0'][1] == "wombat")

    
    def test_changeitpost_basic_iterative_targeted(self):    
        """basic_iterative, targeted algo option, 1st run"""
        # !Attn.: this is a long-running test        
        
        ALGO_NAME = AlteritAlgoName.basic_iterative_algo
        #1st run:
        temp_dict = {}        
        temp_dict['steps'] = 200 #enough to obtain terrier
        #temp_dict['verbose'] = True
        temp_dict['targeted'] = True
        temp_dict['target_goal'] = 189
        self.ffile['alter_parameters'] = json.dumps(temp_dict)
        #print(self.ffile)
        r = self.client.post(f'/alterit/?algo_name={ALGO_NAME}', files = self.ffile)
        self.assertTrue(r.status_code, 200)
        i1, i2, i3, i4 = from_zip_stream_to_att_data(r,
                                                    )        
        self.assertTrue(i1['0'][1] == "Lakeland_terrier")
        


# if __name__ == '__main__':
#     unittest.main()
    
