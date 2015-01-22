
import numpy as np
from pl2mind.datasets import SNP
from pl2mind.research import randomize_snps

def test_randomization(batch_size=5):
    rng = np.random.RandomState([2014,10,31])
    dataset = SNP.MultiChromosome(which_set="train",
                                  chromosomes=5,
                                  dataset_name="snp_1k",
                                  add_noise=True)
    iterator = dataset.iterator(mode=None,
                                batch_size=batch_size,
                                rng=rng,
                                data_specs=dataset.data_specs)

    next_data = iterator.next()
    print float(len(np.where(next_data[0].eval() != dataset.Xs[0][:5])[0].tolist())) /\
        (batch_size * dataset.Xs[0].shape[1])
    for chrom_data in next_data[:-1]:
        for x in chrom_data.eval().flatten():
            assert x in [0, .5, 1], x
