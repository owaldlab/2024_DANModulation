import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


import navis.interfaces.neuprint as neu
from navis.interfaces.neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from navis.interfaces.neuprint import fetch_adjacencies, fetch_synapse_connections

from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
from PyQt5 import QtWidgets, QtCore

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Title
import numpy as np

import pickle
import matplotlib.colors as mcolors
import itertools

import re

## function that retrieves specified connections via neuprint

def loadConnections(typeA="^PAM.*", typeB="^PAM.*", silent=True,bidirectional=False, cached = True):
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
    try:
        #### hackily handles most common case where PAM-PAM or PAM-All connections are requested, loading them from pre-created pickle
        if typeA=="^PAM.*" and typeB=="^PAM.*" and cached:
            with open('pickles/PAM_PAM_Connections.pkl', 'rb') as file:
                PAM_PAM_Connections = pickle.load(file)
            return PAM_PAM_Connections
        if typeA=="^PAM.*" and typeB=="^.*" and cached:
            with open('pickles/PAM_All_Connections.pkl', 'rb') as file:
                PAM_All_Connections = pickle.load(file)
            return PAM_All_Connections
    except Exception:
        print("Warning, no Pickle files found.")
        
    if bidirectional == True:
        neuron_dfAB, conn_dfAB = fetch_adjacencies(NC(status='Traced', type=typeA, regex=True), NC(status='Traced', type=typeB, regex=True))
        conn_dfAB = neu.merge_neuron_properties(neuron_dfAB, conn_dfAB, ['type', 'instance'])
        neuron_dfBA, conn_dfBA = fetch_adjacencies(NC(status='Traced', type=typeB, regex=True), NC(status='Traced', type=typeA, regex=True))
        conn_dfBA = neu.merge_neuron_properties(neuron_dfBA, conn_dfBA, ['type', 'instance'])
        conn_df = pd.concat([conn_dfAB, conn_dfBA]).drop_duplicates()

    if bidirectional == False:
        neuron_df, conn_df = fetch_adjacencies(NC(status='Traced', type=typeA, regex=True), NC(status='Traced', type=typeB, regex=True))
        conn_df = neu.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])

    loadedConnections = conn_df
    #loadedConnections.sort_values('weight', ascending=False, inplace=True)
    if not silent:
        print(neuron_df)
        print(conn_df)

    return loadedConnections


def loadPickle(name, rel_path="pickles/"):
    with open('pickles/'+name+'.pkl', 'rb') as file:
        dataframe = pickle.load(file)
    return dataframe


#### functions to extract inputs and outputs of a specific neuron type, and extract unique connection partners
def extractInputsPerType(target = "PAM05", targetMode = 'type', connections=None):
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
    if targetMode == "type":
        target_pattern = target+r'\_?.?'
        regex=True
    if targetMode == "instance":
        target_pattern = target
        regex=False
    inputs = connections[connections[targetMode+'_post'].str.contains(target_pattern, regex=regex)]
    return inputs

def extractOutputsPerType(target = "PAM05", targetMode = 'type', connections=None):
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
    if targetMode == "type":
        target_pattern = target+r'\_?.?'
        regex=True
    if targetMode == "instance":
        target_pattern = target
        regex=False
    outputs = connections[connections[targetMode+'_pre'].str.contains(target_pattern, regex=regex)]
    return outputs

def listUniqueConnectionPartners(connections, type = "pre", printOut = True):
    typePartners = extractUniqueConnectionPartners(connections,type=type,partnerType="type",mergePAMSubtypes=True)
    subTypePartners = extractUniqueConnectionPartners(connections,type=type,partnerType="type")
    instancePartners = extractUniqueConnectionPartners(connections,type=type,partnerType="instance")
    bodyIdPartners = extractUniqueConnectionPartners(connections,type=type,partnerType="bodyId")
    if printOut: 
        print("Unique connection partners")
        print("by superType")
        print("['PAM']")
        print("by type")
        print(typePartners)
        print("by subType")
        print(subTypePartners)
        print("by instance")
        print(instancePartners)
        print("by bodyId")
        print(bodyIdPartners)
    return subTypePartners, instancePartners, bodyIdPartners


## extracts dataframe of unique connection partners in a dataframe that has connections (e.g. dendrites or axons dataframe)
## takes type : 'pre' or 'post' and partnerType : 'instance' or 'type' or 'bodyId', 'typemerged' acts like type but merges PAM neuron subgroups (e.g. PAM04_a, PAM04_b -> PAM04)
def extractUniqueConnectionPartners(connections, type = "pre", partnerType = "instance",mergePAMSubtypes = False):
    if mergePAMSubtypes:
        connections = connections.replace(to_replace=r'PAM(\d{2})\_?\w?', value=r'PAM\1', regex=True)
    connectionpartners = connections[partnerType+'_'+type].unique()
    return connectionpartners


## takes type : 'pre' or 'post' and partnerMode : 'instance' or 'type' or 'bodyId'
def collapseConnections(connections, type="pre", partnerMode="type",mergePAMSubtypes = False, mergePAMsupertype = False, mergeOthers = True):
    """
    Collapses a connections dataframe by summing the weights of connections.

    Parameters:
    - connections (DataFrame): The dataframe containing synaptic connections.
    - type (str): Specifies the type of connections to consider; either 'pre' for presynaptic or 'post' for postsynaptic.
    - partnerMode (str): Specifies the partner type to consider; options are 'instance', 'type', or 'bodyId'.
    - mergePAMSubtypes (bool, optional): If True, merges PAM neuron subtypes (e.g., PAM04_a, PAM04_b -> PAM04).
    - mergePAMsupertype (bool, optional): If True, merges all PAM neuron types into a single 'PAM' supertype.
    - mergeOthers (bool, optional): If True, merges other neuron types based on predefined patterns.

    Returns:
    - DataFrame: A dataframe with connections collapsed by the specified partnerMode and type, with a sum of the weights.
    """
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
    if mergePAMSubtypes:
        connections = connections.replace(to_replace=r'PAM(\d{2})\_?\w?', value=r'PAM\1', regex=True)
    if mergePAMsupertype:
        #TODO FIX PAM MERGING
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
def extractUniquePartnerConnectionStrengthIterated(targets, targetMode='type', connections=None,type="pre", partnerMode="type",connectionType = "inputs", normalized = True, etc = True, etcTreshhold=0.03,mergePAMSubtypes=False, mergeOthers = True):
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
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
def extractUniquePartnerConnectionStrength(target, targetMode = 'type', connections=None, type="pre", partnerMode="type",connectionType = "inputs", normalized = True, etc = True, etcTreshhold=0.03, mergePAMSubtypes=False, mergePAMsupertype = False, mergeOthers = True):
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
    if connectionType == "inputs":
        conn = extractInputsPerType(target,targetMode=targetMode, connections=connections)
    if connectionType == "outputs":
        conn = extractOutputsPerType(target,targetMode=targetMode,connections=connections)
    targetConnections = collapseConnections(conn, type=type, partnerMode=partnerMode, mergePAMSubtypes = mergePAMSubtypes, mergePAMsupertype = mergePAMsupertype, mergeOthers = mergeOthers)
    targetConnections = targetConnections.rename(columns={'total_weight': target})
    return targetConnections

### functions for plotting synapse statistics as stacked bar charts

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

def getPAMcolors():
    return PAM_colors

import matplotlib.colors as mcolors

def generateColorShades(base_color, num_shades=4, lighten=False):
    # Convert the base color to RGBA
    rgba_color = mcolors.to_rgba(base_color)
    # Convert RGBA to HSV
    hsv_color = mcolors.rgb_to_hsv(rgba_color[:3])
    
    shades = []
    step = (0.8 / num_shades) if lighten else (-0.8 / num_shades)  # Smaller step for more nuanced shade differences
    
    for i in range(num_shades):
        # Modify the value component of the HSV color
        new_v = max(0, min(1, hsv_color[2] + (step * i)))
        new_color = mcolors.hsv_to_rgb([hsv_color[0], hsv_color[1], new_v])
        # Ensure RGB values are within the 0-1 range
        new_color = [max(0, min(1, channel)) for channel in new_color]
        # Convert back to RGBA with original alpha and ensure they are in the correct range
        shades.append((new_color[0], new_color[1], new_color[2], rgba_color[3]))
    
    return shades



MB_rois=["a'L(R)","aL(R)","b'L(R)","bL(R)","gL(R)","CA(L)","a'L(L)","aL(L)","b'L(L)","bL(L)","gL(L)", "CA(R)", "PED(R)"]
non_MB_rois=["CRE(L)", "CRE(R)", "EB", "LAL(R)", "SIP(L)", "SIP(R)", "SLP(R)", "SMP(L)", "SMP(R)", "LAL(L)"]

def getROIs(ROI = "", mode = "array"):
    """
        Returns a specific set of ROIs.
        Parameters:
            (string) ROI: Specify ROI set, e.g. "MB" or "non_MB".
            (string) mode: Specify the formatting of the ROIs. Defaults to "array", an array of strings. "regexOr" returns a single string of all ROIs separated by the '|' character.     
    """
    roi = None
    found = False
    if ROI == "MB":
        rois = MB_rois
        found = True
    if ROI == "non_MB" or ROI == "non MB" or ROI == "non-MB" or ROI == "nonMB":
        rois = non_MB_rois
        found = True
    if isinstance(ROI, list) and mode == "neuronTypesOr":
        found = True
    
    if not found:   
        raise ValueError("Specified ROI set not found. Please specify either 'MB' or 'non_MB' or add additional ROIs in getROIs function.")

    if mode=="array":
        return rois
    if mode == "regexOr":
        roiString = rois[0]
        roiString = re.sub(r'[\(\)]', lambda x: "\\" + x.group(), roiString)
        for roi in rois[1:]:
            roi = re.sub(r'[\(\)]', lambda x: "\\" + x.group(), roi)
            roiString += "|" + roi
        return roiString
    if mode == "neuronTypesOr":
        neuronTypesString = ROI[0]
        neuronTypesString = re.sub(r'[\(\)]', lambda x: "\\" + x.group(), neuronTypesString)
        for neurontype in ROI[1:]:
            neurontype = re.sub(r'[\(\)]', lambda x: "\\" + x.group(), neurontype)
            neuronTypesString += "|" + neurontype
        return neuronTypesString

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
        gui = False
    else:
        gui = True
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
    if gui == False:
        if settingsSpec is not None:
            plt.figtext(0.5, 0.01, settingsSpec, wrap=True, horizontalalignment='center', fontsize=8, color='grey')
            plt.show()


def plotPAMStatistic(targets, targetMode = "type",etcTreshhold=0.03, partnerMode="type", connections=None, normalized=True, title="PAM-PAM Connections Statistic", mergePAMSubtypes=False, mergePAMsupertype=False, mergeOthers=True, yLabel="% of synaptic connections", color_dict=PAM_colors,settingsSpec="", ax = None,weightFilterThreshhold=1):
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
    - mergePAMsupertype: Boolean indicating whether to merge all PAMs.
    - mergeOthers: Boolean indicating whether to merge other neuron types.
    - yLabel: The label for the y-axis.
    - color_dict: Dictionary mapping synapse types to colors.
    - weightFilterThreshhold: Filters out any connection with weight beneath threshhold. Use cautiously as >50% of connections have weight 1 in many cases.
    """
    if weightFilterThreshhold > 1:
        connections = connections[connections['weight']>weightFilterThreshhold]
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
    pamMerged = ""
    if ax is None:
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1 = None
        ax2 = None
    else:
        ax1, ax2 = ax
    if mergePAMSubtypes:
        pamMerged = " (PAM subtypes merged)"
    if not normalized:
        yLabel = "Summed weight of synaptic connections"
    if mergePAMsupertype:
        mergePAMSubtypes = False
    settingsSpec = settingsSpec+f"Target Mode: {targetMode}, Partner Mode: {partnerMode}, Threshold: {etcTreshhold:.4f},  {'Normalized' if normalized else 'Not Normalized'}, Merge PAM subtypes: {'Yes' if mergePAMSubtypes else 'No'}, Merge PAMs: {'Yes' if mergePAMsupertype else 'No'}, Merge Others: {'Yes' if mergeOthers else 'No'}"
    # Extract input connections
    connectionType = "inputs"
    type = "pre"
    visualizeSynapseConnectionTable(
        extractUniquePartnerConnectionStrengthIterated(targets, targetMode=targetMode,connections=connections, connectionType=connectionType, type=type, partnerMode=partnerMode, etcTreshhold=etcTreshhold, normalized=normalized, mergePAMSubtypes=mergePAMSubtypes, mergeOthers=mergeOthers),
        xLabel="PAM type", yLabel=yLabel, title=title, titleSuffix=" - inputs" + pamMerged, color_dict=color_dict, settingsSpec=settingsSpec,ax=ax2)

    # Extract output connections
    connectionType = "outputs"
    type = "post"
    visualizeSynapseConnectionTable(
        extractUniquePartnerConnectionStrengthIterated(targets, targetMode=targetMode, connections=connections, connectionType=connectionType, type=type, partnerMode=partnerMode, etcTreshhold=etcTreshhold, normalized=normalized, mergePAMSubtypes=mergePAMSubtypes, mergeOthers=mergeOthers),
        xLabel="PAM type", yLabel=yLabel, title=title, titleSuffix=" - outputs" + pamMerged, color_dict=color_dict,settingsSpec=settingsSpec,ax=ax1)


### functions for organizing and classifying synaptic connections

def collapseNeuronNames(dataframe, patterns=["KC", "MBON"], targets=["type", "instance"], sides=["pre", "post"], suffix = "s"):
    """
    Collapse neuron names in a dataframe based on specified patterns.

    This function modifies the input dataframe by replacing neuron names that contain
    any of the specified patterns with a shortened name (pattern + 's').

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing neuron data.
    - patterns (list of str): Patterns to match for collapsing neuron names.
    - targets (list of str): Column name parts to apply the patterns to ('type', 'instance').
    - sides (list of str): Suffixes of the column names to which the patterns will be applied ('pre', 'post').

    Returns:
    - pd.DataFrame: The function returns the modified dataframe.
    """

    for pattern in patterns:
        replacement = pattern + suffix
        for target in targets:
            if sides != None:
                for side in sides:
                    dataframe[target + "_" + side] = dataframe[target + "_" + side].replace(
                    to_replace=r'.*' + pattern + r'.*', value=replacement, regex=True)
            else:
                dataframe[target] = dataframe[target].replace(
                to_replace=r'.*' + pattern + r'.*', value=replacement, regex=True)
    return dataframe

def decimateConnections(connections, percentage=10):
    """
    Reduces the number of connections in a DataFrame to a specified percentage by random selection.

    Parameters:
    - connections (pd.DataFrame): The DataFrame containing connection data.
    - percentage (int): The percentage of connections to retain.

    Returns:
    - pd.DataFrame: A new DataFrame containing only the specified percentage of connections.
    """
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100.")

    # Calculate the number of connections to retain
    retain_count = int(len(connections) * (percentage / 100.0))

    # Randomly select a subset of connections
    decimated_connections = connections.sample(n=retain_count)

    return decimated_connections


# by default, yields 0 for MB, 1 for non-MB
def classifySynapseROIs(connections, roiClasses = ["MB","non-MB"], classificationColumnName = "roiClassification", synapseSide = "pre"):
    """
    Classifies synapses based on their region of interest (ROI) into predefined classes.

    This function adds a new column to the 'connections' DataFrame indicating the classification of each synapse
    based on the ROI it belongs to. The classification is determined by checking if the ROI of the synapse
    matches any of the patterns specified in the 'roiClasses' list.

    Parameters:
    - connections (pd.DataFrame): The DataFrame containing synapse data.
    - roiClasses (list of str): List of ROI classes to classify synapses into. Default is ["MB", "non-MB"].
    - classificationColumnName (str): Name of the new column to be added to the DataFrame for storing classification results.
    - synapseSide (str): Specifies whether to classify based on the 'pre' or 'post' side of the synapse. Default is 'pre'.

    Returns:
    - pd.DataFrame: The modified DataFrame with an additional column indicating the ROI classification of each synapse.
    """

    connections = connections.copy()
    connections[classificationColumnName] = 0
    roiClassesTerms = []
    for roiClass in roiClasses:
        roiClassesTerms.append(getROIs(roiClass,mode="regexOr")) ### gets associated list of ROIs in a specified ROI class)
    #print(roiClassesTerms) 
    def get_classification(row):
        if row['roi_'+synapseSide]==None:
            return None
        i = 0
        for roiClassTerm in roiClassesTerms:
            #print(roiClassTerm)
            if re.search(roiClassTerm, row['roi_'+synapseSide]):
                return i
            i=+1
        return None        
    connections[classificationColumnName] = connections.apply(get_classification, axis=1)
    return connections

def classifySynapseConnectivity(connections, types=["PAM", "KC", "MBON"], classificationColumnName = "connectivityClassification"):
    """
    Classify connections based on neuron types, subtypes, instances, and body IDs.

    This function adds a 'classification' column to the 'connections' DataFrame, which
    indicates the level of similarity between the pre- and post-synaptic neurons.
    The classification is an integer where:
    - 0 indicates no match,
    - 1 indicates a match of supertype (e.g., PAM -> PAM),
    - 2 indicates a match of types (e.g., PAM01 -> PAM01),
    - 3 indicates a match of instances (e.g., PAM05(B1ped)_L -> PAM05(B1ped)_L),
    - 4 indicates a match of body IDs.

    Parameters:
    - connections (pd.DataFrame): DataFrame containing connection data with 'type_pre',
      'type_post', 'instance_pre', 'instance_post', 'bodyId_pre', and 'bodyId_post' columns.
    - types (list of str): List of neuron types to check for matches. Default is ["PAM", "KC", "MBON"].

    Returns:
    - pd.DataFrame: The modified DataFrame with an additional 'classification' column.
    """
    connections = connections.copy()
    connections[classificationColumnName] = 0
    def get_classification(row):
        if row['bodyId_pre'] == row['bodyId_post']:
            return 4
        elif row['instance_pre'] == row['instance_post']:
            return 3
        elif row['type_pre'] == row['type_post']:
            return 2
        elif any(row['type_pre'].startswith(t) and row['type_post'].startswith(t) for t in types):
            return 1
        else:
            return 0
    connections[classificationColumnName] = connections.apply(get_classification, axis=1)
    return connections



#### functions related to synapses

def getAnatomicalOutlines(roiName, view='xz'):
    """
    Fetches the mesh for a region of interest (ROI) from neuprint and converts it to 2D outlines.

    Args:
        roiName (str): The name of the region of interest.
        view (str): The view for the 2D projection, default is 'xz'.

    Returns:
        numpy.ndarray: An array containing the 2D outlines of the ROI.
    """
    ##fetch mesh from neuprint
    mesh = neu.fetch_roi(roiName)
    mesh.color = (.9, .9, .9, .75) #,(.9, .9, .9, .05)
    ## convert to 2d outlines
    roi2d = np.array(mesh.to_2d(alpha=2, view=view))
    roi_outlines = np.append(roi2d, np.repeat(mesh.center[2], roi2d.shape[0]).reshape(roi2d.shape[0], 1), axis=1)
    return roi_outlines

def plotSynapseConfidence(synapseTable, mode="pre", show=True):
    """
    Plots a histogram of synapse confidences and prints the fifth percentile.

    This function takes a DataFrame containing synapse data, plots a histogram of the
    confidence values for either presynaptic or postsynaptic neurons, and calculates
    the fifth percentile of these confidence values.

    Parameters:
    - synapseTable (pd.DataFrame): DataFrame containing synapse data with 'confidence_pre' or 'confidence_post'.
    - mode (str): Specifies whether to use 'pre' for presynaptic or 'post' for postsynaptic confidence values. Default is 'pre'.
    - show (bool): If True, the histogram will be displayed. Default is True.

    Returns:
    - float: The fifth percentile of the specified confidence values.
    """
    # Plotting histogram of frequency of each value in column 'confidence_pre' or 'confidence_post'
    plt.hist(synapseTable['confidence_'+mode], bins=50)
    fifth_percentile = np.percentile(synapseTable['confidence_'+mode], 5)
    print("The fifth percentile of the 'confidence_"+mode+"' values is:", fifth_percentile)

    plt.xlabel('Confidence '+mode)
    plt.ylabel('Frequency')
    plt.title('Histogram of Confidence '+mode+' Values')
    if show:
        plt.show()
    return fifth_percentile

def filterSynapseConfidence(synapseTable, mode="pre", percentile = 5):
    perc = np.percentile(synapseTable['confidence_'+mode], percentile)
    synapseTable = synapseTable[synapseTable['confidence_'+mode] > perc]
    return synapseTable

def plotSynapseGroups(synapseTables, title="Synapse Plot", colors=["red", "blue", "green", "yellow"], ROIoutlines=None, ROIname="ROI outline", coordinates=["x", "z"], showPlot=True):
    """
    Plots groups of synapses with different colors on a 2D plot.

    Args:
        synapseTables (list of pd.DataFrame): A list of DataFrames where each contains synapse data to be plotted.
        title (str): The title of the plot.
        colors (list of str): A list of colors for each group of synapses.
        ROIoutlines (numpy.ndarray): An array containing the 2D outlines of the ROI to be plotted.
        ROIname (str): The label for the ROI outlines in the legend.
        coordinates (list of str): The coordinate names to be used for plotting, default is ["x", "z"].
        showPlot (bool): If True, the plot will be displayed.

    Returns:
        None
    """
    print("This function will become deprecated, please use plotClassifiedSynapses instead.")

    p = figure(title=title)
    i = 0
    for synapses in synapseTables:
        p.scatter(synapses[coordinates[0]+'_post'], synapses[coordinates[1]+'_post'], color=colors[i])
        i = i+1
    p.y_range.flipped = True
    if ROIoutlines is not None:
        p.scatter(ROIoutlines[:,0], ROIoutlines[:,1], legend_label=ROIname)
    if showPlot:
        show(p)



def plotSynapseClassification(synapseTable, title="Synapse Classification", classificationColumn="connectivityClassification", classificationInterval=[0, 1, 2, 3, 4], colors=["grey", "red", "blue", "green", "cyan"], labels=["Heterogenous", "Same Supertype", "Same Type", "Same Instance", "Same Body ID"], ROIoutlines=None, ROIname="ROI outline", coordinates=["x", "z"], showPlot=True, subtitle=None):
    """
    Plots synapses classified by similarity on a 2D plot with different colors for each classification category.

    Args:
        synapseTable (pd.DataFrame): A DataFrame containing synapse data with a classification column.
        title (str): The title of the plot.
        classificationColumn (str): The name of the column in synapseTable that contains classification data.
        classificationInterval (list of int): A list of classification categories to be plotted.
        colors (list of str): A list of colors for each classification category.
        labels (list of str): A list of labels for each classification category.
        ROIoutlines (numpy.ndarray): An array containing the 2D outlines of the ROI to be plotted.
        ROIname (str): The label for the ROI outlines in the legend.
        coordinates (list of str): The coordinate names to be used for plotting, default is ["x", "z"].
        showPlot (bool): If True, the plot will be displayed.
        subtitle (str): An optional subtitle for the plot.

    Returns:
        None
    """
    print("This function will become deprecated, please use plotClassifiedSynapses instead.")
    p = figure(title=title)
    i = 0
    for classification in classificationInterval:
        synapses = synapseTable[(synapseTable[classificationColumn] == classification)]
        n = synapses.index.size
        p.scatter(synapses[coordinates[0]+'_post'], synapses[coordinates[1]+'_post'], color=colors[i], legend_label=f"{labels[i]}, {n}")
        i += 1
    p.y_range.flipped = True
    if ROIoutlines is not None:
        p.scatter(ROIoutlines[:,0], ROIoutlines[:,1], legend_label=ROIname)
    if subtitle:
        p.add_layout(Title(text=subtitle, align="center"), "below")
    if showPlot:
        show(p)


def plotPAMTypePresynapses(synapseTables, pamType="PAM01", roi_outlines=None, roiName="MB",classificationColumnName = "connectivityClassification"):
    """
    Plots the classification of presynapses for a specified PAM neuron type.

    This function processes synapse tables to filter and classify presynapses of a given PAM type,
    then plots the classification results using two different coordinate systems.

    Args:
        synapseTables (list of pd.DataFrame): A list of DataFrames containing synapse data.
        pamType (str): The PAM neuron type to be plotted. Defaults to "PAM01".
        roi_outlines (list of numpy.ndarray): A list containing the 2D outlines of the ROI to be plotted.
            Defaults to None, which will be interpreted as [None, None].
        roiName (str): The label for the ROI outlines in the legend. Defaults to "MB".

    Returns:
        None
    """
    print("This function will become deprecated, please use plotClassifiedSynapses instead.")
    collapsedPAMpresynapses = collapseNeuronNames(synapseTables.copy(), [pamType], ["type"], sides=["pre", "post"], suffix="")
    filteredPAMpresynapses = collapsedPAMpresynapses[collapsedPAMpresynapses["type_pre"] == pamType].copy()
    filteredPAMpresynapses.drop(columns=classificationColumnName, inplace=True)
    filteredPAMpresynapses = classifySynapseConnectivity(filteredPAMpresynapses,classificationColumnName=classificationColumnName)
    title = pamType + " Presynapses Classification"
    if roi_outlines is None:
        roi_outlines = [None, None]
    plotSynapseClassification(filteredPAMpresynapses, title=title, subtitle="Same Supertype: " + pamType + "-PAM, Same Type: " + pamType + "-" + pamType, ROIoutlines=roi_outlines[0], ROIname=roiName)
    plotSynapseClassification(filteredPAMpresynapses, title=title, coordinates=["x", "y"], ROIoutlines=roi_outlines[1], ROIname=roiName)


### multiple inputGroups not implemented yet, please pass array of just one input group for now
def plotClassifiedSynapses(inputGroups, title="Synapse Classification", inputGroupColors=[None],
                           filterConfig={
                               'classificationColumns': ['roiClassification', 'connectivityClassification'],
                               'bounds': [2, 5],
                               'directional': [False, False],
                               'direction': [None, None],
                               'colorScheme': ['#c6cfd0', '#ADD8E6', '#87CEEB', '#4682B4', '#000080', '#e2d3d5', '#FA8072', '#FF6347', '#FF0000', '#8B0000'],
                               'classificationLabels': [
                                   ["MB", "non-MB"],
                                   ["Heterogenous", "Same Supertype", "Same Type", "Same Instance", "Same Body ID"]
                               ]}, 
                               ROIoutlines=None, ROIname="ROI outline", 
                               coordinates=["x", "z"], 
                               showPlot=True, subtext=None, showLegend = True, showTitle=True,
                               ax = None,
                               dotSize = 4):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        gui = False
    else:
        gui = True
        fig = ax.figure
    

    ###update subtext
    subtext = subtext + " Classification Columns: ["
    for col in filterConfig['classificationColumns']:
        subtext = subtext + col+","
    subtext = subtext + "]"

    if subtext:
        ax.text(0.5, -0.1, subtext, fontsize=10, color='grey', ha='center', transform=ax.transAxes)

    # for now, only single inputs are supported
    inputSynapses = inputGroups[0]

    classificationGroupIndex = 0
    # iterates through all combinations of classification dimensions and plots all synapses that fall within it
    totalN = 0
    for classificationIndices in itertools.product(*[range(bound) for bound in filterConfig['bounds']]):
        queryString = ""
        for i, classificationColumn in enumerate(filterConfig['classificationColumns']):
            if i != 0:
                queryString += " and "
            queryString += f"{classificationColumn} == {classificationIndices[i]}"
        #print(queryString)
        synapses = inputSynapses.query(queryString)
        n = synapses.index.size
        totalN = totalN + n
        # plot synapses in classification group
        ax.scatter(
            synapses[coordinates[0] + '_pre'], synapses[coordinates[1] + '_pre'],
            color=filterConfig['colorScheme'][classificationGroupIndex],
            label=f"{filterConfig['classificationLabels'][0][classificationIndices[0]] + ', ' + filterConfig['classificationLabels'][1][classificationIndices[1]]}, {n}",
            s = dotSize
        )

        classificationGroupIndex += 1

    if ROIoutlines is not None:
        ax.scatter(ROIoutlines[:, 0], ROIoutlines[:, 1], color='black', label=ROIname, s=1)
    if showTitle:
        ax.set_title(f"{title}, N={totalN}")

    if showLegend:
        ax.legend(loc='upper right', markerscale=3)
    ax.set_xlabel(coordinates[0])
    ax.set_ylabel(coordinates[1])
    ax.invert_yaxis()
    if gui == False and showPlot == True:
        plt.show()
