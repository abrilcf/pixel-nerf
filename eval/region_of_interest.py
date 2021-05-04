import numpy as np
import torch
import cv2


def structure_of_interest(struct_type, slices, blur_radius, pixel_range):
    """preserves pixels value like bone structure
    :input: 
        struct_type - string (e.g., 'bone' / 'organ')
        slices - ndarray shape=(#slices,H,W), 
        blur_radius - int hypeparameter needs to be odd number
        pixel_range - float the greater the more 'layers' of 
                        bone it will preserve (e.g., 0.1)
    :return: 
        ndarray shape=(#slices,H,W)
    """
    # so pixels value are averaged by neighbours
    # apply a Gaussian blur to a copy of a centred slice
    # then find the area with the largest intensity value (e.g bone)
    n_centred_slice = int(slices.shape[0]/2)
    original = slices[n_centred_slice:n_centred_slice+1,:,:]
    centred_slice = original.copy()
    print("centred_slice: ", centred_slice.shape)
    centred_slice = cv2.GaussianBlur(centred_slice, (blur_radius, blur_radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(centred_slice.squeeze())

    # preserve pixels of structure of interest otherwise go numb
    for s in range(slices.shape[0]):
        for h in range(slices.shape[1]):
            for w in range(slices.shape[2]):
                # pixel_rane=0.085 seems to work fine to highlight bone structure in THIS case
                if struct_type == 'bone':
                    if slices[s,h,w] < maxVal-pixel_range:
                        slices[s,h,w] = 0
                elif struct_type == 'organ':
                    if slices[s,h,w] < (maxVal-(pixel_range*4)) or slices[s,h,w] > (maxVal-(pixel_range*1.38)):
                        slices[s,h,w] = 0


    return slices
