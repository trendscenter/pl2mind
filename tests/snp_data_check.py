import numpy as np
from os import path

prior_tolerance=0.1
snp_dir = "/export/research/analysis/human/collaboration/SzImGen/IMPUTE/forsergey/d_MCIC+COBRE1_realdata/chr1/"
synth_dir = "/export/research/analysis/human/collaboration/SzImGen/IMPUTE/forsergey/d_MCIC+COBRE1_synthetic_n2000/chr1_synthetic/"

controls_file = path.join(synth_dir, "chr1_risk_n1000.controls.gen")
controls = open(controls_file, "r")

cases_file = path.join(synth_dir, "chr1_risk_n1000.cases.gen")
cases = open(cases_file, "r")

subject_file = path.join(snp_dir, "chr1_original.tped")
subjects = open(subject_file, "r")

bim_file = path.join(snp_dir, "chr1_original.bim")
bims = open(bim_file, "r")
bim_lines = bims.readlines()
minor_majors = {}
for line in bim_lines:
    parsed = line.split("\t")
    parsed = [p.translate(None, "\n") for p in parsed]
    name = parsed[1]
    minor = parsed[4]
    major = parsed[5]
    assert minor in "TCAG", "Bad minor %s." % minor
    assert major in "TCAG", "Bad major %s." % major 
    assert minor + major in "TTCCT" or "AAGGA"
    minor_majors[name] = (minor, major)
"""
leg_file = path.join(snp_dir, "chr1.leg")
leg = open(leg_file, "r")
for line in leg.readlines():
    parsed = line.split(" ")
    parsed = [p.translate(None, "\n") for p in parsed]
    name = parsed[0]
    if name in ["rsID", "rsdummy"]:
        continue
    minor = parsed[2]
    major = parsed[3]
    assert minor_majors[name] == (minor, major)
"""

subject_lines = subjects.readlines()
subject_values = {}
for line in subject_lines:
    parsed = line.split(" ")
    name = parsed[1]
    minor, major = minor_majors[name]
    snp_values = parsed[4:]
    snp_values = [s.translate(None, "\n") for s in snp_values]
    values = []
    for i in range(0, len(snp_values), 2):
        A, B = snp_values[i], snp_values[i+1]
        if A == minor:
            A = 0
        elif A == major:
            A = 1
        else:
            raise ValueError("%s." % A)
        if B == minor:
            B = 0
        elif B == major:
            B = 1
        else:
            raise ValueError("%s." % B)
        values.append(A + B)
    subject_values[name] = values

controls_lines = controls.readlines()
controls_values = {}
for line in controls_lines:
    parsed = line.split(" ")
    name = parsed[1]
    minor = parsed[3]
    major = parsed[4]
    if name == "rsdummy":
        continue
    assert minor_majors[name] == (minor, major),\
        "Minor and major do not match %r vs %r" % ((minor, major), minor_majors[name])
    values = [int(i) for i in parsed[5:]]
    assert len(values) % 3 == 0
    controls_values[name] = [values[j:j+3].index(1) for j in range(0, len(values), 3)]

cases_lines = cases.readlines()
cases_values = {}
for line in cases_lines:
    parsed = line.split(" ")
    name = parsed[1]
    if name == "rsdummy":
        continue
    values = [int(i) for i in parsed[5:]]
    assert len(values) % 3 == 0
    cases_values[name] = [values[j:j+3].index(1) for j in range(0, len(values), 3)]

for name in controls_values:
    assert name in cases_values
    assert name in subject_values, "SNP not found in subjects: %s" % name

misses = 0
for j, name in enumerate(controls_values):
    controls_value = controls_values[name]
    cases_value = cases_values[name]
    subject_value = subject_values[name]
    
    controls_probs = [controls_value.count(i) * 1. / len(controls_value) for i in range(3)]
    cases_probs = [cases_value.count(i) * 1. / len(cases_value) for i in range(3)]
    subject_probs = [subject_value.count(i) * 1. / len(subject_value) for i in range(3)]
    
    if not np.allclose(controls_probs, cases_probs, atol=prior_tolerance) or\
            not np.allclose(controls_probs, subject_probs, atol=prior_tolerance):
        misses += 1
        print "****WARNING****\nSNP prior probabilities on snp %d(%s) do not match (P(0), P(1), P(2)):\n controls:\t%r\n cases:\t\t%r\n subjects:\t%r"\
            % (j, name, controls_probs, cases_probs, subject_probs)

    if misses > 10:
        raise ValueError("Too many priors off.")
    

