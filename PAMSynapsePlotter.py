from IPython.display import display, clear_output


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


import navis.interfaces.neuprint as neu
from navis.interfaces.neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from navis.interfaces.neuprint import fetch_adjacencies, fetch_synapse_connections

from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
from PyQt5 import QtWidgets, QtCore

import re

from utils import loadConnections, plotSynapseClassification, filterSynapseConfidence, classify_connections, get_outlines, plotPAMTypePresynapses


## setting up Neuprint Client
## using dotenv to import Janelia PAT
from dotenv import load_dotenv
import os
load_dotenv()

## fetching Janelia client
client = neu.Client('https://neuprint.janelia.org/', dataset='hemibrain:v1.2.1', token=os.environ.get("JANELIA_PAT"))



### get PAM-PAM connections
filteredConnections = loadConnections(silent=False)

### retrieving all connections between PAM and non-PAM neurons
filteredPAMConnections = loadConnections("^PAM.*","^.*", minWeight=1,bidirectional=True)


# Define the possible targets by type
possibleTargetsType = {
    "PAM01": "PAM01",
    "PAM02": "PAM02",
    "PAM03": "PAM03",
    "PAM04": "PAM04",
    "PAM05": "PAM05",
    "PAM06": "PAM06",
    "PAM07": "PAM07",
    "PAM08": "PAM08",
    "PAM09": "PAM09",
    "PAM10": "PAM10",
    "PAM11": "PAM11",
    "PAM12": "PAM12",
    "PAM13": "PAM13",
    "PAM14": "PAM14",
    "PAM15": "PAM15",
}

## fetch all synapses that involve PAM,
## postsynapses are 1:1 identical to presynapses, both have n=150049 
pam_criteria = NC(status='Traced', type="^PAM.*",regex=True)
all_criteria = NC(status='Traced', type="^.*",regex=True)

allPAMpresynapses_criteria = SC(type='pre', primary_only=True)
#allPAMpostsynapses_criteria = SC(type='post', primary_only=True)
allPAMpresynapses = fetch_synapse_connections(pam_criteria, all_criteria, allPAMpresynapses_criteria,batch_size=10000)
#allPAMpostsynapses = fetch_synapse_connections(pam_criteria, all_criteria, allPAMpostsynapses_criteria,batch_size=10000)

all_neurons, roi_count = neu.fetch_neurons(NC(status='Traced',type="^.*",regex=True)) 

PAMpresynapses = filterSynapseConfidence(allPAMpresynapses,mode="pre",percentile=10) 
mergedPAMpresynapses = neu.merge_neuron_properties(all_neurons,PAMpresynapses,properties=["type", "instance"])
#mergedPAMpostynapses = neu.merge_neuron_properties(all_neurons,allPAMpostsynapses,properties=["type", "instance"])

mergedPAMpresynapses = classify_connections(mergedPAMpresynapses)
#mergedPAMpostynapses = classify_connections(mergedPAMpostynapses)

MB_outlines_xz = get_outlines("MB")
MB_outlines_xy = get_outlines("MB","xy")

title="PAM Presynapses Classification"
decimated_connections = mergedPAMpresynapses.sample(n=10000)
plotSynapseClassification(decimated_connections, title=title, coordinates=["x","z"],ROIoutlines=MB_outlines_xz,ROIname="MB")
plotSynapseClassification(decimated_connections, title=title, coordinates=["x","y"],ROIoutlines=MB_outlines_xy,ROIname="MB")

plotPAMTypePresynapses(mergedPAMpresynapses,pamType="PAM15",roi_outlines=[MB_outlines_xz,MB_outlines_xy],roiName="MB")
