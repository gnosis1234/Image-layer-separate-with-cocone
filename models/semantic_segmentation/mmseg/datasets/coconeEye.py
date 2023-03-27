# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CoconeEye(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    # CLASSES = ['Background', 'L_eyeball_H', 'L_eyelid1_B', 'L_eyelid1_F', 'L_eyelid2_F', 'L_eyelid3_F', 'R_eyeball_H', 'R_eyelid1_B', 'R_eyelid1_F', 'R_eyelid2_F', 'R_eyelid3_F']
    # CLASSES = ['Background', 'L_eyeball_H', 'L_eyelid1_B', 'L_eyelid', 'R_eyeball_H', 'R_eyelid1_B', 'R_eyelid']
    CLASSES = ['Background', 'eyeball_H', 'eyelid1_B', 'eyelid_F']
    PALETTE = [[0, 0, 0],       # Background
                    [255, 0, 255],   # L_eyeball_H
                    [0, 192, 0],     # L_eyelid1_B
                    # [196, 196, 196], # L_eyelid1_F
                    # # [190, 153, 153], # L_eyelid2_F
                    [0, 128, 255],   # L_eyelid3_F
                    # [255, 128, 0],   # R_eyeball_H
                    # [102, 20, 30],   # R_eyelid1_B
                    # [128, 64, 255],  # R_eyelid1_F
                    # [255, 255, 0],   # R_eyelid2_F
                    # [0, 255, 255],   # R_eyelid3_F
                ]       
    def __init__(self, **kwargs):
        super(CoconeEye, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            # result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=False,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
