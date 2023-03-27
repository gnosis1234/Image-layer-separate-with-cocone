

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

class Dense_CRF2D():
    def __init__(self):
        # dense CRF
        self.MAX_ITER = 3
        self.POS_W = 3
        self.POS_XY_STD = 1
        self.Bi_W = 4
        self.Bi_XY_STD = 15
        self.Bi_RGB_STD = 3


    def forward(self, img, output_probs):
        c = output_probs.shape[0]
        h = output_probs.shape[1]
        w = output_probs.shape[2]
        # img = np.resize(img, (h,w, 3))
        U = utils.unary_from_softmax(output_probs)
        U = np.ascontiguousarray(U)

        img = np.ascontiguousarray(img)

        d = dcrf.DenseCRF2D(w, h, c)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.POS_XY_STD, compat=self.POS_W)
        d.addPairwiseBilateral(sxy=self.Bi_XY_STD, srgb=self.Bi_RGB_STD, rgbim=img, compat=self.Bi_W)

        Q = d.inference(self.MAX_ITER)
        Q = np.array(Q).reshape((c, h, w))
        return Q

    def dense_crf_wrapper(self, args):
        return self.forward(args[0], args[1])
