from skimage.segmentation import clear_border
import pytesseract
import uvicorn
import numpy as np
import cv2

import json, os, zipfile, io


from alpm.hardcode import *
from alpm.lp_util.base_class import AlpmError


def get_class_idx(label):
    """obtain imagenet id by label"""
    labelPath = os.path.join(os.path.dirname(__file__),
                             "imagenet_class_index.json")

    with open(labelPath) as f:
        imageNetClasses = {labels[1]: int(idx) for (idx, labels) in
                           json.load(f).items()}

    return imageNetClasses.get(label, None)


def demonstrate_image(title, image, waitKey=True):
    """ former debug_imshow """
    cv2.imshow(title, image)
    if waitKey:
        cv2.waitKey(0)


def get_input_par_from_request(input_image=None,
                               input_image_path=None,
                               image_parameters={}):
    """
        rearrange input **kwargs
        """
    t_input_image_path = input_image_path.decode('utf-8')
    t_image_parameters = json.loads(image_parameters.decode('utf-8').replace("'", "\""))

    input_image1 = input_image.file.read()
    input_image.file.close()
    nparr = np.frombuffer(input_image1, np.uint8)
    t_input_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return t_input_image, t_input_image_path, t_image_parameters


def preprocess_image_for_resnet50(image):
    """
    swap color channels, resize the input image, and add a batch dimension
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    return image


def make_zip_from_all_data(all_data: tuple):
    """makes zipfile "on the fly" and saves to disk """

    s = io.BytesIO()
    temp_dict = {}
    with zipfile.ZipFile(s, mode="w") as zf:
        for i, data_chunk in enumerate(all_data):
            img_name = "chunk_{:02d}".format(i + 1)  # hardcoded
            if type(data_chunk) is np.ndarray:
                temp_obj = data_chunk.tobytes()
                if img_name == 'chunk_02':
                    temp_dict[img_name] = data_chunk.shape
            elif type(data_chunk) is dict:
                temp_obj = json.dumps(data_chunk).encode()
            else:
                raise AlpmError(WRONG_FILE_DATATYPE_ERROR)
            zf.writestr(img_name, temp_obj)
        zf.writestr('npshapes', json.dumps(temp_dict).encode())  # hardcoded

    return s


async def a_make_zip_from_all_data(all_data: tuple):
    """makes zipfile "on the fly" and saves to disk """

    s = io.BytesIO()
    temp_dict = {}
    with zipfile.ZipFile(s, mode="w") as zf:
        for i, data_chunk in enumerate(all_data):
            img_name = "chunk_{:02d}".format(i + 1)  # hardcoded
            if type(data_chunk) is np.ndarray:
                temp_obj = data_chunk.tobytes()
                if img_name == 'chunk_02':
                    temp_dict[img_name] = data_chunk.shape
            elif type(data_chunk) is dict:
                temp_obj = json.dumps(data_chunk).encode()
            else:
                raise AlpmError(WRONG_FILE_DATATYPE_ERROR)
            zf.writestr(img_name, temp_obj)
        zf.writestr('npshapes', json.dumps(temp_dict).encode())  # hardcoded

    return s


def from_zip_stream_to_att_data(r,
                                target_path=None,
                                ):
    """
    this foo takes response object with zip file and extract attack data
    please, refer to:
    - alpm.changeit.tweaks.BasicIterativeAttack
    AND/OR:
    - alpm.alpmapi.get_altit_algo
    specifications.

    - r :: response object, media_type="application/x-zip-compressed"
    - target_path - path to save file, default is None (do not save)

    - return: tuple of (back_data, deltaNp, adverImage, delta_image, npshapes),
    where:
            - back_data (chunk_01):: regognition data for attacked image (i.e. fake recognitions)
            - deltaNp (chunk_02):: np.array :: delta addition to image (attack add-on), shape the same as image, type np.float32
            - adverImage (chunk_03):: np.array :: attackedge, type np.uint8
            - delta_image (chunk_04):: np.array :: this is a non-strict transformation of deltaNp, for visualising goal only, type np.uint8
            - npshapes (npshapes):: serialized dict like {"chunk_02": [224, 224, 3]} with a shape of adverImage (other np.arrays have the same shape)
                            this foo version output is NOT serializable
    """

    if r is None:
        raise ValueError('response object must not be None')
        return tuple([None] * 5)

    try:
        temp_content = r.content
    except:
        temp_content = r.getvalue()
    #    raise ValueError('Can not read response content')

    if target_path is not None:
        with open(target_path, 'wb') as ff:
            ff.write(temp_content)

    temp_data = {}
    y = io.BytesIO()
    y.write(temp_content)
    myzipfile = zipfile.ZipFile(y)
    for name in myzipfile.namelist():
        temp_data[name] = myzipfile.open(name).read()
    back_shape_size = json.loads(temp_data['npshapes'].decode()).get('chunk_02')
    
    return json.loads(temp_data['chunk_01'].decode()),\
        np.frombuffer(temp_data['chunk_02'], dtype=np.float32).reshape(back_shape_size),\
        np.frombuffer(temp_data['chunk_03'], dtype=np.uint8).reshape(back_shape_size),\
        np.frombuffer(temp_data['chunk_04'], dtype=np.uint8).reshape(back_shape_size)


# async def zip_file_gen(myzipfile):
#    for name in myzipfile.namelist():
#        yield name


async def a_from_zip_stream_to_att_data(r,
                                        target_path=None,
                                        ):
    """

    """

    if r is None:
        raise ValueError('response object must not be None')
        return tuple([None] * 5)

    try:
        temp_content = r.content
    except:
        raise ValueError('Can not read response content')

    if target_path is not None:
        async with open(target_path, 'wb') as ff:
            await ff.write(temp_content)

    temp_data = {}
    y = io.BytesIO()
    y.write(temp_content)
    myzipfile = zipfile.ZipFile(y)
    for name in myzipfile.namelist():
        temp_data[name] = myzipfile.open(name).read()
    #async for name in zip_file_gen(myzipfile):
        #temp_struct = myzipfile.open(name)
        #temp_data[name] = await temp_struct.read()

    back_shape_size = json.loads(temp_data['npshapes'].decode()).get('chunk_02')
    return json.loads(temp_data['chunk_01'].decode()),\
        np.frombuffer(temp_data['chunk_02'], dtype=np.float32).reshape(back_shape_size),\
        np.frombuffer(temp_data['chunk_03'], dtype=np.uint8).reshape(back_shape_size),\
        np.frombuffer(temp_data['chunk_04'], dtype=np.uint8).reshape(back_shape_size)

# def locate_license_plate_candidates(gray,
#                                     keep = MAX_AREAS_NUMBER,
#                                     first_kernel = LARGE_LP_IMAGE_KERNEL_SIZE):
# 	""" find regions with LP numbers"""

# 	rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, first_kernel_size)
# 	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

# ############################################################################################
# 	# next, find regions in the image that are light
# 	squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# 	light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
# 	light = cv2.threshold(light, 0, 255,
# 		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 	self.debug_imshow("Light Regions", light)

# 		# compute the Scharr gradient representation of the blackhat
# 		# image in the x-direction and then scale the result back to
# 		# the range [0, 255]
# 		gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
# 			dx=1, dy=0, ksize=-1)
# 		gradX = np.absolute(gradX)
# 		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# 		gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
# 		gradX = gradX.astype("uint8")
# 		self.debug_imshow("Scharr", gradX)

# 		# blur the gradient representation, applying a closing
# 		# operation, and threshold the image using Otsu's method
# 		gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
# 		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
# 		thresh = cv2.threshold(gradX, 0, 255,
# 			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 		self.debug_imshow("Grad Thresh", thresh)

# 		# perform a series of erosions and dilations to cleanup the
# 		# thresholded image
# 		thresh = cv2.erode(thresh, None, iterations=2)
# 		thresh = cv2.dilate(thresh, None, iterations=2)
# 		self.debug_imshow("Grad Erode/Dilate", thresh)

# 		# take the bitwise AND between the threshold result and the
# 		# light regions of the image
# 		thresh = cv2.bitwise_and(thresh, thresh, mask=light)
# 		thresh = cv2.dilate(thresh, None, iterations=2)
# 		thresh = cv2.erode(thresh, None, iterations=1)
# 		self.debug_imshow("Final", thresh, waitKey=True)

# 		# find contours in the thresholded image and sort them by
# 		# their size in descending order, keeping only the largest
# 		# ones
# 		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 			cv2.CHAIN_APPROX_SIMPLE)
# 		cnts = cnts[0]
# 		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

# 		# return the list of contours
# 		return cnts
