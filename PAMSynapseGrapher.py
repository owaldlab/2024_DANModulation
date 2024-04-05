import ipywidgets as widgets
from IPython.display import display, clear_output

### imports
import navis
import fafbseg
import flybrains

import numpy as np
import seaborn as sns
import itertools
import pandas as pd
from tqdm import tqdm
from functools import reduce
from tabulate import tabulate
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import rgb2hex, to_rgb
import matplotlib.gridspec as gridspec
import hvplot.pandas
from bokeh.plotting import figure, show, output_notebook
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import ipywidgets

import scipy
import networkx as nx

import IProgress

import pyroglancer
from pyroglancer.localserver import startdataserver, closedataserver
from pyroglancer.flywire import flywireurl2dict, add_flywirelayer, set_flywireviewerstate

import navis.interfaces.neuprint as neu
from navis.interfaces.neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from navis.interfaces.neuprint import fetch_adjacencies, fetch_synapse_connections

from pyroglancer.layers import create_nglayer, setlayerproperty
from pyroglancer.ngviewer import openviewer, closeviewer,setviewerstate, get_ngscreenshot
from pyroglancer.ngspaces import create_ngspace
from pyroglancer.createconfig import createconfig

from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import re

import gspread


## setting up Neuprint Client
## using dotenv to import Janelia PAT
from dotenv import load_dotenv
import os
load_dotenv()

## fetching Janelia client
client = neu.Client('https://neuprint.janelia.org/', dataset='hemibrain:v1.2.1', token=os.environ.get("JANELIA_PAT"))


## function that retrieves specified connections via neuprint

def getFilteredConnections(typeA="^PAM.*", typeB="^PAM.*", minWeight=1, silent=True):
    """
    Retrieves connections between neurons of specified types with a minimum synaptic weight from the Neuprint database.

    Args:
        typeA (str): Regular expression pattern for the type of pre-synaptic neurons. Defaults to "^PAM.*".
        typeB (str): Regular expression pattern for the type of post-synaptic neurons. Defaults to "^PAM.*".
        minWeight (int): The minimum synaptic weight for connections to be retrieved. Defaults to 1.
        silent (bool): If False, the function will print the dataframes containing neuron and connection information. Defaults to True.

    Returns:
        pandas.DataFrame: A dataframe containing the filtered connections with a weight greater than or equal to minWeight.
    """
    neuron_df, conn_df = fetch_adjacencies(NC(status='Traced', type=typeA, regex=True), NC(status='Traced', type=typeB, regex=True))
    conn_df = neu.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])
    filteredConnections = conn_df[conn_df['weight'] >= minWeight]
    filteredConnections.sort_values('weight', ascending=False, inplace=True)
    if not silent:
        print(neuron_df)
        print(conn_df)

    return filteredConnections


### get PAM-PAM connections
filteredConnections = getFilteredConnections(silent=False)
### retrieving all connections between PAM and non-PAM neurons
filteredPAMConnections = getFilteredConnections("^PAM.*","^.*", minWeight=1)


#### functions to extract axons and dendrites of a specific neuron type, and extract unique connection partners

### extracts dendrites and axons for a specific PAM type and returns them as a dataframe
def extractDendritesPerType(target = "PAM05", targetMode = 'type', connections=filteredConnections):
    if targetMode == "type":
        target_pattern = target+r'\_?.?'
        regex=True
    if targetMode == "instance":
        target_pattern = target
        regex=False
    dendrites = connections[connections[targetMode+'_post'].str.contains(target_pattern, regex=regex)]
    #print(dendrites)
    return dendrites

### extracts dendrites and axons for a specific PAM type and returns them as a dataframe
def extractAxonsPerType(target = "PAM05", targetMode = 'type', connections=filteredConnections):
    if targetMode == "type":
        target_pattern = target+r'\_?.?'
        regex=True
    if targetMode == "instance":
        target_pattern = target
        regex=False
    axons = connections[connections[targetMode+'_pre'].str.contains(target_pattern, regex=regex)]
    #print(axons)
    return axons


## takes type : 'pre' or 'post' and partnerMode : 'instance' or 'type' or 'bodyId'
def collapseConnections(connections, type="pre", partnerMode="type",mergePAMSubtypes = False, mergePAMs = False, mergeOthers = True):
    """
    Collapses a connections dataframe by summing the weights of connections.

    Parameters:
    - connections (DataFrame): The dataframe containing synaptic connections.
    - type (str): Specifies the type of connections to consider; either 'pre' for presynaptic or 'post' for postsynaptic.
    - partnerMode (str): Specifies the partner type to consider; options are 'instance', 'type', or 'bodyId'.
    - mergePAMSubtypes (bool, optional): If True, merges PAM neuron subtypes (e.g., PAM04_a, PAM04_b -> PAM04).
    - mergePAMs (bool, optional): If True, merges all PAM neuron types into a single 'PAM' category.
    - mergeOthers (bool, optional): If True, merges other neuron types based on predefined patterns.

    Returns:
    - DataFrame: A dataframe with connections collapsed by the specified partnerMode and type, with a sum of the weights.
    """
    if mergePAMSubtypes:
        connections = connections.replace(to_replace=r'PAM(\d{2})\_?\w?', value=r'PAM\1', regex=True)
    if mergePAMs:
        print("PAM merging not working yet. Proceeding unmerged.") 
        #connections['type_' + type] = connections['type_' + type].replace(to_replace=r'PAM(\d{2})\_?\w?', value=r'PAM', regex=True)
    if mergeOthers:
        connections.loc[:, 'type_' + type] = connections['type_' + type].replace(to_replace=r'.*KC.*', value=r'KCs', regex=True)
        connections.loc[:, 'type_' + type] = connections['type_' + type].replace(to_replace=r'.*MBON.*', value=r'MBONs', regex=True)
    # Group the connections by the specified partnerMode and type, then sum the weights
    grouped = connections.groupby([partnerMode + '_' + type]).agg({'weight': 'sum'}).reset_index()
    # Rename the columns to reflect the collapsed data
    grouped.columns = [partnerMode + '_' + type, 'total_weight']
    return grouped.sort_values('total_weight', ascending=False)

### for each target in targets, this extracts weights of synaptic connections in each neuron type called 'target'
def extractUniquePartnerConnectionStrengthIterated(targets, targetMode='type', connections=filteredConnections,type="pre", partnerMode="type",connectionType = "dendrites", normalized = True, etc = True, etcTreshhold=0.03,mergePAMSubtypes=False, mergeOthers = True):
    connectionsTable = pd.DataFrame()
    for target in targets:
        targetConnections = extractUniquePartnerConnectionStrength(target, targetMode=targetMode, connections=connections, type=type, partnerMode=partnerMode,connectionType = connectionType, normalized = normalized, etc = etc, etcTreshhold=0.03, mergePAMSubtypes=mergePAMSubtypes, mergeOthers=mergeOthers)
        if connectionsTable.empty:
            connectionsTable = targetConnections
        else:
            connectionsTable = connectionsTable.merge(targetConnections, on=partnerMode+'_'+type, how='outer')
        connectionsTable = connectionsTable.fillna(0)
        connectionsTable = connectionsTable.set_index(partnerMode+'_'+type)
    
    if etc:
        threshold = etcTreshhold * connectionsTable.sum().sum()
        others = connectionsTable[connectionsTable.sum(axis=1) < threshold].sum()
        connectionsTable = connectionsTable[connectionsTable.sum(axis=1) >= threshold]
        if not connectionsTable.empty:
            connectionsTable.loc['others'] = others
        else:
            # Initialize connectionsTable with the correct columns if it's empty
            connectionsTable = pd.DataFrame(others).T
            connectionsTable.index = ['others']

    if normalized:
        connectionsTable = connectionsTable.div(connectionsTable.sum(axis=0), axis=1) * 100

    return connectionsTable

### extract weights of synaptic connections per types in the neuron type called 'target'
def extractUniquePartnerConnectionStrength(target, targetMode = 'type', connections=filteredConnections, type="pre", partnerMode="type",connectionType = "dendrites", normalized = True, etc = True, etcTreshhold=0.03, mergePAMSubtypes=False, mergePAMs = False, mergeOthers = True):
    if connectionType == "dendrites":
        conn = extractDendritesPerType(target,targetMode=targetMode, connections=connections)
    if connectionType == "axons":
        conn = extractAxonsPerType(target,targetMode=targetMode,connections=connections)
    targetConnections = collapseConnections(conn, type=type, partnerMode=partnerMode, mergePAMSubtypes = mergePAMSubtypes, mergePAMs = mergePAMs, mergeOthers = mergeOthers)
    targetConnections = targetConnections.rename(columns={'total_weight': target})
    return targetConnections

### functions for plotting synapse statistics

PAM_colors = {
    '.*PAM01.*': '#FF6F61',  # Living Coral
    '.*PAM02.*': '#6B5B95',  # Wisteria
    '.*PAM03.*': '#88B04B',  # Greenery
    '.*PAM04.*': '#F7CAC9',  # Rose Quartz
    '.*PAM05.*': '#92A8D1',  # Serenity
    '.*PAM06.*': '#964F4C',  # Marsala
    '.*PAM07.*': '#B565A7',  # Radiant Orchid
    '.*PAM08.*': '#009B77',  # Emerald
    '.*PAM09.*': '#DD4124',  # Flame
    '.*PAM10.*': '#45B8AC',  # Turquoise
    '.*PAM11.*': '#EFC050',  # Sunflower
    '.*PAM12.*': '#5B5EA6',  # Blue Bell
    '.*PAM13.*': '#9B2335',  # Red Dahlia
    '.*PAM14.*': '#DFCFBE',  # Almond Milk
    '.*PAM15.*': '#BC243C',  # Blue Atoll
    '.*other.*' : 'grey',  # Thistle
    '.*KC.*': '#55B4B0',     # Scarlet Red
    '.*MBON.*': '#C3447A',    # Pink Peacock
    '.*APL.*':'#FFD700',  # Gold
    '.*DPM.*':'#A52A2A',  # Brown
}

def visualizeSynapseConnectionTable(connectionTable, title="PAM-PAM Synapse Statistic", titleSuffix="", xLabel="None", yLabel='None', color_dict=PAM_colors, legend_title="", settingsSpec=None, ax = None):
    """
    Visualizes a synapse connection table as a stacked bar chart.

    Parameters:
    - connectionTable: DataFrame containing the synapse connection data.
    - title: The title of the plot.
    - titleSuffix: Additional suffix to append to the plot title.
    - xLabel: The label for the x-axis.
    - yLabel: The label for the y-axis.
    - color_dict: Dictionary mapping synapse types to colors.
    - legend_title: The title for the legend.
    """
    connectionTable = connectionTable.T
    # Plot initialization
    # Use the provided ax for plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    bar_width = 0.85  # Width of each bar

    # Iterative plotting
    for bar_index, (bar_name, row_values) in enumerate(connectionTable.iterrows()):
        bottom_height = 0  # Starting point for the first segment of each bar
        for segment_name, segment_value in row_values.sort_values(ascending=False).items():
            color = next((v for k, v in color_dict.items() if re.search(k, segment_name)), None)
            ax.bar(bar_index, segment_value, color=color, edgecolor='gray', linewidth=1, bottom=bottom_height, label=segment_name if bar_index == 0 else "", width=bar_width)
            bottom_height += segment_value  # Update the bottom for the next segment in this bar

    # Customizing the plot with labels, title, etc.
    ax.set_xticks(range(len(connectionTable)))
    ax.set_xticklabels(connectionTable.index)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title + titleSuffix)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))
    # Fine-tuning appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plotPAMStatistic(targets, targetMode = "type",etcTreshhold=0.03, partnerMode="type", connections=filteredConnections, normalized=True, title="PAM-PAM Connections Statistic", mergePAMSubtypes=False, mergePAMs=False, mergeOthers=True, yLabel="% of synaptic connections", color_dict=PAM_colors,settingsSpec="", ax = None):
    """
    Plots PAM synapse statistics for a given set of targets.

    Parameters:
    - targets: List of target neuron types to include in the analysis.
    - targetMode: Whether each bar should be graphed for 'type' or for 'instance' groups.
    - etcTreshhold: Threshold for including 'etc' category in the analysis.
    - partnerMode: The type of partner neuron to consider in the analysis.
    - connections: DataFrame containing the synapse connection data.
    - normalized: Boolean indicating whether to normalize the connection strengths.
    - title: The title of the plot.
    - mergePAMSubtypes: Boolean indicating whether to merge PAM subtypes.
    - mergePAMs: Boolean indicating whether to merge all PAMs.
    - mergeOthers: Boolean indicating whether to merge other neuron types.
    - yLabel: The label for the y-axis.
    - color_dict: Dictionary mapping synapse types to colors.
    """
    pamMerged = ""
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        ax1, ax2 = ax
    if mergePAMSubtypes:
        pamMerged = " (PAM subtypes merged)"
    if not normalized:
        yLabel = "Summed weight of synaptic connections"
    if mergePAMs:
        mergePAMSubtypes = False
    settingsSpec = settingsSpec+f"Target Mode: {targetMode}, Partner Mode: {partnerMode}, Threshold: {etcTreshhold:.4f},  {'Normalized' if normalized else 'Not Normalized'}, Merge PAM subtypes: {'Yes' if mergePAMSubtypes else 'No'}, Merge PAMs: {'Yes' if mergePAMs else 'No'}, Merge Others: {'Yes' if mergeOthers else 'No'}"
    # Extract dendrite connections for PAMs 01-06
    connectionType = "dendrites"
    type = "pre"
    visualizeSynapseConnectionTable(
        extractUniquePartnerConnectionStrengthIterated(targets, targetMode=targetMode,connections=connections, connectionType=connectionType, type=type, partnerMode=partnerMode, etcTreshhold=etcTreshhold, normalized=normalized, mergePAMSubtypes=mergePAMSubtypes, mergeOthers=mergeOthers),
        xLabel="PAM type", yLabel=yLabel, title=title, titleSuffix=" - dendritic inputs" + pamMerged, color_dict=color_dict, settingsSpec=settingsSpec,ax=ax1)

    # Extract axon connections
    connectionType = "axons"
    type = "post"
    visualizeSynapseConnectionTable(
        extractUniquePartnerConnectionStrengthIterated(targets, targetMode=targetMode, connections=connections, connectionType=connectionType, type=type, partnerMode=partnerMode, etcTreshhold=etcTreshhold, normalized=normalized, mergePAMSubtypes=mergePAMSubtypes, mergeOthers=mergeOthers),
        xLabel="PAM type", yLabel=yLabel, title=title, titleSuffix=" - axonal outputs" + pamMerged, color_dict=color_dict,settingsSpec=settingsSpec,ax=ax2)



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

# Define the possible targets by subtype
possibleTargetsSubtype = {
    "PAM01_a": "PAM01_a",
    "PAM01_b": "PAM01_b",
    "PAM02": "PAM02",
    "PAM03": "PAM03",
    "PAM04_a": "PAM04_a",
    "PAM04_b": "PAM04_b",
    "PAM05": "PAM05",
    "PAM06_a": "PAM06_a",
    "PAM06_b": "PAM06_b",
    "PAM07": "PAM07",
    "PAM08_a": "PAM08_a",
    "PAM08_b": "PAM08_b",
    "PAM08_c": "PAM08_c",
    "PAM08_d": "PAM08_d",
    "PAM08_e": "PAM08_e",
    "PAM09": "PAM09",
    "PAM10": "PAM10",
    "PAM11": "PAM11",
    "PAM12": "PAM12",
    "PAM13": "PAM13",
    "PAM14": "PAM14",
    "PAM15_a": "PAM15_a",
    "PAM15_b": "PAM15_b"
}

possibleTargetsInstance = {
    "PAM01_a(y5)_L": "PAM01_a(y5)_L",
    "PAM01_a(y5)_R": "PAM01_a(y5)_R",
    "PAM01_b(y5)_L": "PAM01_b(y5)_L",
    "PAM01_b(y5)_R": "PAM01_b(y5)_R",
    "PAM02(B'2a)_L": "PAM02(B'2a)_L",
    "PAM02(B'2a)_R": "PAM02(B'2a)_R",
    "PAM03(B2B'2a)_L": "PAM03(B2B'2a)_L",
    "PAM03(B2B'2a)_R": "PAM03(B2B'2a)_R",
    "PAM04_a(B2)_L": "PAM04_a(B2)_L",
    "PAM04_a(B2)_R": "PAM04_a(B2)_R",
    "PAM04_b(B2)(ADM02)_L": "PAM04_b(B2)(ADM02)_L",
    "PAM04_b(B2)_R": "PAM04_b(B2)_R",
    "PAM05(B'2p)_L": "PAM05(B'2p)_L",
    "PAM05(B'2p)_R": "PAM05(B'2p)_R",
    "PAM06_a(B'2m)_L": "PAM06_a(B'2m)_L",
    "PAM06_a(B'2m)_R": "PAM06_a(B'2m)_R",
    "PAM06_b(B'2m)_L": "PAM06_b(B'2m)_L",
    "PAM06_b(B'2m)_R": "PAM06_b(B'2m)_R",
    "PAM07(y4<y1y2)_L": "PAM07(y4<y1y2)_L",
    "PAM07(y4<y1y2)_R": "PAM07(y4<y1y2)_R",
    "PAM08_a(y4)_L": "PAM08_a(y4)_L",
    "PAM08_a(y4)_R": "PAM08_a(y4)_R",
    "PAM08_b(y4)_L": "PAM08_b(y4)_L",
    "PAM08_b(y4)_R": "PAM08_b(y4)_R",
    "PAM08_c(y4)_L": "PAM08_c(y4)_L",
    "PAM08_c(y4)_R": "PAM08_c(y4)_R",
    "PAM08_d(y4)_L": "PAM08_d(y4)_L",
    "PAM08_d(y4)_R": "PAM08_d(y4)_R",
    "PAM08_e(y4)_L": "PAM08_e(y4)_L",
    "PAM08_e(y4)_R": "PAM08_e(y4)_R",
    "PAM09(B1ped)_L": "PAM09(B1ped)_L",
    "PAM09(B1ped)_R": "PAM09(B1ped)_R",
    "PAM10(B1)_L": "PAM10(B1)_L",
    "PAM10(B1)_R": "PAM10(B1)_R",
    "PAM11(a1)_L": "PAM11(a1)_L",
    "PAM11(a1)_R": "PAM11(a1)_R",
    "PAM12(y3)_L": "PAM12(y3)_L",
    "PAM12(y3)_R": "PAM12(y3)_R",
    "PAM13(B'1ap)_L": "PAM13(B'1ap)_L",
    "PAM13(B'1ap)_R": "PAM13(B'1ap)_R",
    "PAM14(B'1m)_L": "PAM14(B'1m)_L",
    "PAM14(B'1m)_R": "PAM14(B'1m)_R",
    "PAM15_a(y5B'2a)_L": "PAM15_a(y5B'2a)_L",
    "PAM15_b(y5B'2a)_R": "PAM15_b(y5B'2a)_R"
}



from PyQt5 import QtWidgets, QtCore

# Create the main window class
class PAMSynapseGrapherWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle('PAM Synapse Grapher')
        self.setGeometry(100, 100, 800, 600)

        # Create the central widget and layout
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Create a horizontal layout for the main window
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Create a vertical layout for the settings
        settings_layout = QtWidgets.QVBoxLayout()


        # Create two matplotlib figures and canvases to display them with increased size
        self.figure1 = mpl.figure.Figure(figsize=(10, 8))
        self.canvas1 = FigureCanvasQTAgg(self.figure1)

        self.figure2 = mpl.figure.Figure(figsize=(10, 8))
        self.canvas2 = FigureCanvasQTAgg(self.figure2)


        # Create the UI components

        # Visualization Mode Radio Buttons
        self.vis_mode_radio = QtWidgets.QButtonGroup()
        vis_mode_label = QtWidgets.QLabel("Visualization Mode:")
        pam_pam_synapses_radio = QtWidgets.QRadioButton('PAM-PAM synapses')
        all_pam_synapses_radio = QtWidgets.QRadioButton('All PAM synapses')
        self.vis_mode_radio.addButton(pam_pam_synapses_radio, id=0)
        self.vis_mode_radio.addButton(all_pam_synapses_radio, id=1)
        pam_pam_synapses_radio.setChecked(True)

        # Target Mode Radio Buttons
        self.target_mode_radio = QtWidgets.QButtonGroup()
        target_mode_label = QtWidgets.QLabel("Target Mode:")
        type_radio = QtWidgets.QRadioButton('type')
        instance_radio = QtWidgets.QRadioButton('instance')
        self.target_mode_radio.addButton(type_radio, id=0)
        self.target_mode_radio.addButton(instance_radio, id=1)
        type_radio.setChecked(True)

        # Targets List Widget
        self.targets_list_widget = QtWidgets.QListWidget()
        self.targets_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.targets_list_widget.addItems(possibleTargetsType.keys())

        # Partner Type Radio Buttons
        self.partner_type_radio = QtWidgets.QButtonGroup()
        partner_type_label = QtWidgets.QLabel("Partner Type:")
        type_partner_radio = QtWidgets.QRadioButton('type')
        instance_partner_radio = QtWidgets.QRadioButton('instance')
        self.partner_type_radio.addButton(type_partner_radio, id=0)
        self.partner_type_radio.addButton(instance_partner_radio, id=1)
        type_partner_radio.setChecked(True)

        # Merge PAM Types Checkbox
        self.merge_pam_types_checkbox = QtWidgets.QCheckBox('Merge PAM types')
        self.merge_pam_types_checkbox.setChecked(True)

        # Etc Threshold Float Text
        self.etc_threshhold_floattext = QtWidgets.QDoubleSpinBox()
        self.etc_threshhold_floattext.setValue(0.02)
        self.etc_threshhold_floattext.setSuffix(' Etc Threshhold')

        # Normalized Checkbox
        self.normalized_checkbox = QtWidgets.QCheckBox('Normalized')

        # Update Plot Button
        self.update_button = QtWidgets.QPushButton('Update Plot')
        self.update_button.clicked.connect(self.update_plot)

        # Add the settings widgets to the settings layout with titles
        settings_layout.addWidget(vis_mode_label)
        settings_layout.addWidget(pam_pam_synapses_radio)
        settings_layout.addWidget(all_pam_synapses_radio)
        settings_layout.addWidget(target_mode_label)
        settings_layout.addWidget(type_radio)
        settings_layout.addWidget(instance_radio)
        settings_layout.addWidget(self.targets_list_widget)
        settings_layout.addWidget(partner_type_label)
        settings_layout.addWidget(type_partner_radio)
        settings_layout.addWidget(instance_partner_radio)
        settings_layout.addWidget(self.merge_pam_types_checkbox)
        settings_layout.addWidget(self.etc_threshhold_floattext)
        settings_layout.addWidget(self.normalized_checkbox)
        settings_layout.addWidget(self.update_button)
        # Create a widget to hold the settings layout and add it to the main layout
        settings_widget = QtWidgets.QWidget()
        settings_widget.setLayout(settings_layout)
        main_layout.addWidget(settings_widget)

        # Create a vertical layout for the plots
        plots_layout = QtWidgets.QVBoxLayout()

        # Add the canvases to the plots layout
        plots_layout.addWidget(self.canvas1)
        plots_layout.addWidget(self.canvas2)

        # Add the plots layout to the main layout
        main_layout.addLayout(plots_layout)

        # Set the main layout as the central widget's layout
        central_widget.setLayout(main_layout)


        # Connect signals
        self.target_mode_radio.buttonClicked.connect(self.update_targets_options)

    # Function to update the targets list widget options based on target mode
    def update_targets_options(self):
        if self.target_mode_radio.checkedId() == 0:  # 'type'
            self.targets_list_widget.clear()
            self.targets_list_widget.addItems(possibleTargetsType.keys())
        elif self.target_mode_radio.checkedId() == 1:  # 'instance'
            self.targets_list_widget.clear()
            self.targets_list_widget.addItems(possibleTargetsInstance.keys())

    # Function to call plotPAMStatistic with the selected values
    def update_plot(self):
        vis_mode = 'PAM-PAM synapses' if self.vis_mode_radio.checkedId() == 0 else 'All PAM synapses'
        conn = filteredPAMConnections if vis_mode == 'All PAM synapses' else filteredConnections
        settings_spec = "Visualization mode: " + vis_mode + ", "
        selected_targets = [item.text() for item in self.targets_list_widget.selectedItems()]

        # Clear the previous figures
        self.figure1.clear()
        self.figure2.clear()

        ax1 = self.figure1.add_subplot(111)
        ax2 = self.figure2.add_subplot(111)
        
        ax = [ax1,ax2]

        plotPAMStatistic(
            connections=conn,
            targets=selected_targets,
            title=vis_mode,
            targetMode='type' if self.target_mode_radio.checkedId() == 0 else 'instance',
            etcTreshhold=self.etc_threshhold_floattext.value(),
            partnerMode='type' if self.partner_type_radio.checkedId() == 0 else 'instance',
            mergePAMSubtypes=self.merge_pam_types_checkbox.isChecked(),
            normalized=self.normalized_checkbox.isChecked(),
            settingsSpec=settings_spec,
            ax=ax
        )
        self.figure1.subplots_adjust(left=0.15, right=0.8)  # Adjust these values as needed
        self.figure2.subplots_adjust(left=0.15, right=0.8)  # Adjust these values as needed

        self.canvas1.draw()  # Render the plot
        self.canvas2.draw()  # Render the plot


# Create and show the application
app = QtWidgets.QApplication([])
window = PAMSynapseGrapherWindow()
window.show()
app.exec_()
