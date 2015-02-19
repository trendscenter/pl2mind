"""
Module for testing MRI generation functionality.
"""

import numpy as np
from os import path

from pl2mind.datasets import MRI_generation
from pl2mind.tools import mri_analysis


class TestMRIGen:
    def __init__(self):
        out_dir = path.join(path.abspath(path.dirname(__file__)), "outs")
        self.source_file = path.join(out_dir, "ica_sources.npy")
        self.mixing_file = path.join(out_dir, "ica_mixing.npy")
        self.nifti_file = path.join(out_dir, "ica_image.nii")
        self.pdf_file = path.join(out_dir, "ica_montage.pdf")
        self.rng = np.random.RandomState([2014,10,31])
        self.mri = None

    def test_build(self):
        self.mri = MRI_generation.MRI_Gen("train", 60, 10, apply_mask=True,
                                     dataset_name="smri",
                                     source_file=self.source_file,
                                     mixing_file=self.mixing_file)
        return self.mri

    def test_ica(self):
        # Difficult to control order of tests...
        if self.mri is None:
            self.mri = self.test_build()
        mri = self.mri
        ica_sources = mri.S
        ica_mixing = mri.A
        np.save(self.source_file, ica_sources)
        np.save(self.mixing_file, ica_mixing)
        ica_nifti = mri_analysis.get_nifti(mri, ica_sources, out_file=self.nifti_file)
        mri_analysis.save_nii_montage(ica_nifti, self.nifti_file, self.pdf_file)

    def test_iteration(self, batch_size=5):
        if self.mri is None:
            self.mri = self.test_build()
        mri = self.mri
        iterator = mri.iterator(mode=None,
                                batch_size=batch_size,
                                rng=self.rng,
                                data_specs=mri.data_specs)

        next_data = iterator.next()
        print next_data
        assert next_data[0].shape == (batch_size, mri.X.shape[1]), (
            "Data is coming out the wrong shape (%s vs %s)"
            % (next_data[0].shape, (batch_size, mri.X.shape[1]))
        )