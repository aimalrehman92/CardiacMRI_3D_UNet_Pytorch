

import numpy as np

########## Pad zero slices front and back ##########

def pad_zero_slices_front_and_back(SA_image, slice_count):

    l1 = round(slice_count/2)

    slices_to_pad = np.zeros((256, 256, slice_count))

    result_ = np.concatenate((slices_to_pad[:,:,:l1], SA_image), axis=2) #back

    result_ = np.concatenate((result_, slices_to_pad[:,:,l1:]), axis=2) #front

    return result_

