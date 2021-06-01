""" base classes fot alpm modules init """


import os, argparse, json, warnings
import uvicorn
import cv2
import numpy as np

from pathlib import Path

from alpm.hardcode import *


class AlpmApiCiCd:
    """
    class for CI/CD
    """

    def __init__(self,
                name =  APPNAME,
                host = DEPLOY_HOST,
                port = RUN_PORT,
                reload = True,
                debug = False):
        """
        obtain host and port for CI/CD
        TODO: terminate code
        """
        self.name = name
        self.debug = debug
        self.reload = reload 
        self.host = host if host else None #TODO: substitute correct code
        self.port = int(port) if port else None #TODO: substitute correct code

    def run(self, debug = False):
        """
        run method for alpmapi app
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=FutureWarning)
        uvicorn.run(self.name+":app", host=self.host, port=self.port, reload=self.reload, debug=self.debug)

    
class AlpmError(Exception):
	"""
	specific numbered error of Discovery class
	"""
	def __init__(self,				
				discover_error_description : str,):
		self.expression = discover_error_description


class BaseAlpm:
	"""
	different nethods to discover LP number.

		Inputs:
		- lp_image: license plate image, cleaned and processed
		- lp_image_hint : license plate number , "recognised" by User, if any
		- kwargs - all other parameters as a dictionary (for FastApi validation)
	"""	
	base_path = Path(__file__).resolve(strict=True).parent.parent.parent
	
	def __init__(self, 
				input_image = None,
				input_image_path = None,
				**kwargs:dict):
		if input_image is not None:
			self.image = input_image
			self.input_image_path = None			
		elif input_image is None and input_image_path is not None:
			self.image_path = os.path.join(self.base_path, input_image_path)
			self.image = cv2.imread(self.image_path)
		else:
			raise AlpmError(NON_INIT_ERRORMESSAGE)
		for every_key, every_value in kwargs.items():
			setattr(self, every_key, every_value)	
	
	def get_visualization(self,
						  image =  None,
						  title =  None,
						  wait_key = True,
						  destroy_the = False, 
						  destroy_all = False):
		"""
		visualize image
		"""
		cv2.imshow(title, image) if image else cv2.imshow(title, self.lp_image)
			# check to see if we should wait for a keypress
		if wait_key:
			cv2.waitKey(0)
		if destroy_the:
			cv2.destroyWindow(title)
		elif destroy_all:
			cv2.destroyAllWindows()