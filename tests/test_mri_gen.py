"""
Module for testing MRI generation functionality.
"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import pylab as plt

import numpy as np
from os import path

from pl2mind.datasets import MRI_generation
from pl2mind.tools import mri_analysis

from pylearn2.utils.rng import make_np_rng
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import sharedX
import theano
from theano import tensor as T


def exp_act_str(exp, act):
    return ("Expected (shape=%r):\n%r\n"
            "Actual (shape=%r):\n%r"
            % (exp.shape, exp,
               act.shape, act))


class TestMRIGen:
    def __init__(self):
        out_dir = path.join(path.abspath(path.dirname(__file__)), "outs")
        self.source_file = path.join(out_dir, "ica_sources.npy")
        self.mixing_file = path.join(out_dir, "ica_mixing.npy")
        self.nifti_file = path.join(out_dir, "ica_image.nii")
        self.pdf_file = path.join(out_dir, "ica_montage.pdf")
        self.hist_plot = path.join(out_dir, "rejective_hist.pdf")
        self.mri = None
        self.mri_with_real = None

    def test_build(self, components=10, use_real=False):
        mri = MRI_generation.MRI_Gen("train", components,
                                          apply_mask=True,
                                          use_real=use_real,
                                          dataset_name="smri",
                                          source_file=self.source_file,
                                          mixing_file=self.mixing_file)
        return mri

    def test_build_with_real(self, components=10):
        mri_with_real = self.test_build(components=components, use_real=True)
        return mri_with_real

    def test_ica(self, components=10):
        # Difficult to control order of tests...
        if self.mri is None:
            self.mri = self.test_build(components=components)
        mri = self.mri
        ica_sources = mri.S
        ica_mixing = mri.A
        np.save(self.source_file, ica_sources)
        np.save(self.mixing_file, ica_mixing)
        ica_nifti = mri_analysis.get_nifti(mri, ica_sources,
                                           out_file=self.nifti_file)
        mri_analysis.save_nii_montage(ica_nifti, self.nifti_file, self.pdf_file)

    def test_iteration(self, components=10, batch_size=30, use_real=False):
        if use_real:
            if self.mri_with_real is None:
                self.mri_with_real = self.test_build_with_real(
                    components=components)
            mri = self.mri_with_real
        else:
            if self.mri is None:
                self.mri = self.test_build(components=components)
            mri = self.mri
        iterator = mri.iterator(mode=None,
                                batch_size=batch_size,
                                data_specs=mri.data_specs)
        next_data = iterator.next()
        labels = next_data[1]
        assert labels.shape == (next_data[0].shape[0], 2), (
            "Labels do not have correct shape (%s vs %s)"
            % (labels.shape, (next_data[0].shape[0], 2))
        )

        if use_real:
            end0 = next_data[0].shape[0] // 4
            end1 = next_data[0].shape[0] // 2
        else:
            end0 = next_data[0].shape[0] // 2
            end1 = next_data[0].shape[0]
        assert np.all(labels[:end0] == [1, 0]), labels[:end0]
        assert np.all(labels[end0:end1] == [0, 1])

        assert next_data[0].shape[1] == mri.X.shape[1]

        for i in range(20):
            print i, mri.X.shape[0] - (i * batch_size)
            try:
                next_data = iterator.next()

                if use_real:
                    end0 = next_data[0].shape[0] // 4
                    end1 = next_data[0].shape[0] // 2
                else:
                    end0 = next_data[0].shape[0] // 2
                    end1 = next_data[0].shape[0]

                labels = next_data[1]
                assert labels.shape == (next_data[0].shape[0], 2), (
                    "Labels do not have correct shape on iteration "
                    "%s (%s vs %s)"
                    % (i, labels.shape, (next_data[0].shape[0], 2))
                )
                assert np.all(labels[:end0] == [1, 0])
                assert np.all(labels[end0:end1] == [0, 1])
            except StopIteration:
                pass

    def test_iteration_with_real(self, components=10, batch_size=10):
        self.test_iteration(components=components, batch_size=batch_size,
                            use_real=True)

    def test_rejective_sample(self, components=10, batch_size=1000):
        if self.mri is None:
            self.mri = self.test_build(components=components)
        mri = self.mri

        A = mri.A
        S = mri.S

        # Get the data info from the mri_gen class
        data_path, label_path = mri.resolve_dataset(mri.which_set, mri.dataset_name)

        # Balance the classes
        y = np.atleast_2d(np.load(label_path)).T
        num_classes = np.amax(y) + 1
        class_counts = [(y == i).sum()
                        for i in range(num_classes)]
        min_count = min(class_counts)
        balanced_idx = []
        for i in range(num_classes):
            idx = [idx for idx, j in enumerate(y) if j == i][:min_count]
            balanced_idx += idx
        balanced_idx.sort()
        assert len(balanced_idx) / min_count == num_classes
        assert len(balanced_idx) % min_count == 0

        y = y[balanced_idx]
        for i in range(num_classes):
            assert (y == i).sum() == min_count

        idx0 = [i for i, j in enumerate(y) if j == 0]
        idx1 = [i for i, j in enumerate(y) if j == 1]
        print idx0
        print idx1

        model0 = [np.histogram(a, density=True) for a in A[idx0].T]
        model1 = [np.histogram(a, density=True) for a in A[idx1].T]
        print model0
        model_complete = [np.histogram(a, density=True) for a in A.T]

        tr1 = make_theano_rng(100, which_method="uniform")
        tr2 = make_theano_rng(100, which_method="uniform")

        generator = MRI_generation.MRI_Generator(A, S, y,
                                                 np.zeros((mri.X.shape[1],)),
                                                 theano_rng=tr1)

        X = mri.X[:batch_size]

        assert np.all(generator.y == y), exp_act_str(y, generator.y)

        assert np.all(generator.hist_set0.eval() == [h for h, e in model0]), (
            exp_act_str(np.array([h for h, e in model0]),
                        np.array(generator.hist_set0.eval())))
        assert np.all(generator.hist_set1.eval() == [h for h, e in model1]), (
            exp_act_str(np.array([h for h, e in model1]),
                        np.array(generator.hist_set1.eval())))
        assert np.all(generator.edge_set0.eval() == [e for h, e in model0]), (
            exp_act_str(np.array([e for h, e in model0]),
                        np.array(generator.edge_set0.eval())))
        assert np.all(generator.edge_set1.eval() == [e for h, e in model1]), (
            exp_act_str(np.array([e for h, e in model1]),
                        np.array(generator.edge_set1.eval())))

        edges = [e for h, e in model0][0]
        h = [h for h, e in model0][0]

        tr1 = make_theano_rng(100, which_method="uniform")
        tr2 = make_theano_rng(100, which_method="uniform")

        es_act = tr1.uniform(
            (batch_size,),
            low=generator.edge_set0[0][0],
            high=generator.edge_set0[0][-1])
        es_exp = tr2.uniform(low=edges[0],
                             high=edges[-1],
                             size=(batch_size,)).eval()

        ec_act, updates = theano.scan(generator.edge_compare,
                                      outputs_info=None,
                                      sequences=[es_act],
                                      non_sequences=[generator.edge_set0[0]])

        ec_exp = [(e > edges[1:]).argmin() for e in es_exp]
        ec_act_e = ec_act.eval()

        #print np.array(ec_exp), ec_act_e
        assert np.all(ec_act_e == ec_exp), exp_act_str(np.array(ec_exp),
                                                            ec_act_e)

        tr1 = make_theano_rng(100, which_method="uniform")
        tr2 = make_theano_rng(100, which_method="uniform")

        es_act = tr1.uniform(
            (batch_size * 20,),
            low=generator.edge_set0[0][0],
            high=generator.edge_set0[0][-1])
        es_exp = tr2.uniform(low=edges[0],
                             high=edges[-1],
                             size=(batch_size * 20,)).eval()
        ec_exp = [(e > edges[1:]).argmin() for e in es_exp]

        us_act = tr1.uniform((batch_size * 20,))
        us_exp = tr2.uniform(size=(batch_size * 20,)).eval()

        tests, updates = theano.scan(generator.hist_compare,
                                     outputs_info=None,
                                     sequences=[sharedX(us_exp),
                                                generator.hist_set0[0][ec_exp]])
        hc_act = es_exp[tests.nonzero()[0].eval()]
        hc_exp = [e for e, u in zip(es_exp, us_exp)
                  if u <= h[(e > edges[1:]).argmin()]]
        hc_act_e = hc_act#.eval()
        assert np.all(hc_act_e == hc_exp), exp_act_str(np.array(hc_exp),
                                                       np.array(hc_act_e))

        def get_column(h, edges, samples, rng):
            h = h * np.diff(edges)
            # es ~ U(min,max)
            es = rng.uniform(low=edges[0],
                             high=edges[-1],
                             size=(samples,)).eval()
            # us ~ U(0,1)
            us = rng.uniform(size=(samples,)).eval()
            # Keep accepted samples
            column = [e for e, u in zip(es,us)
                      if u <= h[(e > edges[1:]).argmin()]]
            return np.array(column)

        def rejectiveSample(model, target_size, samples, rng):
            newA= []
            # For each column
            for ii in range(len(model)):
                h, edges = model[ii]
                column = get_column(h, edges, samples, rng)
                newA.append(column)
            A2 = np.array([a[:target_size] for a in newA]).T
            return(A2)

        tr1 = make_theano_rng(100, which_method="uniform")
        tr2 = make_theano_rng(100, which_method="uniform")
        generator.rng = tr1

        out = generator.get_column(generator.hist_set0[0],
                                   generator.edge_set0[0],
                                   batch_size * 20, batch_size // 2)

        column_act = out[0][0].eval()
        column_exp = get_column(h, edges,
                                batch_size * 20, tr2)[:batch_size // 2]

        #print column_exp, column_act
        assert np.all(column_exp == column_act), exp_act_str(column_exp,
                                                             column_act)

        tr1 = make_theano_rng(100, which_method="uniform")
        tr2 = make_theano_rng(100, which_method="uniform")
        generator.rng = tr1

        A0_exp = rejectiveSample(model0, batch_size,
                                 batch_size * 20, tr2)

        [A0_act, br, es, us, indices], updates = theano.scan(generator.get_column,
                                             outputs_info=[None, None, None, None, None],
                                             sequences=[generator.hist_set0,
                                                        generator.edge_set0],
                                             non_sequences=[batch_size * 20,
                                                            batch_size])

        A0_act = A0_act.T.eval()

        A_comp = rejectiveSample(model_complete, batch_size,
                                 batch_size * 20, tr2)
        print plt.hist(A_comp[:, 0])

        f = plt.figure()
        plt.subplot(2,1,1)
        plt.hist(A0_act[:, 0])
        plt.subplot(2,1,2)
        plt.hist(A0_exp[:, 0])
        print A0_exp.shape
        print plt.hist(A0_exp[:, 0])
        f.savefig(self.hist_plot)
        assert False
        for i, (a_act, a_exp) in enumerate(zip(A0_act.T, A0_exp.T)):
            print i
            #print A0_act[0]
            #print A0_exp[0]
            assert np.all(a_act == a_exp), exp_act_str(a_exp, a_act)
        assert False
        actual = generator.perform(X)
        expected = np.concatenate([A0_exp.dot(S), A1_exp.dot(S)])

        assert np.all(actual == expected), exp_act_str(expected, actual)
