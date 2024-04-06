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


#### functions to extract axons and dendrites of a specific neuron type, and extract unique connection partners

### extracts dendrites and axons for a specific PAM type and returns them as a dataframe
def extractDendritesPerType(target = "PAM05", targetMode = 'type', connections=None):
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
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
def extractAxonsPerType(target = "PAM05", targetMode = 'type', connections=None):
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
    if targetMode == "type":
        target_pattern = target+r'\_?.?'
        regex=True
    if targetMode == "instance":
        target_pattern = target
        regex=False
    axons = connections[connections[targetMode+'_pre'].str.contains(target_pattern, regex=regex)]
    #print(axons)
    return axons

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
def extractUniquePartnerConnectionStrengthIterated(targets, targetMode='type', connections=None,type="pre", partnerMode="type",connectionType = "dendrites", normalized = True, etc = True, etcTreshhold=0.03,mergePAMSubtypes=False, mergeOthers = True):
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
def extractUniquePartnerConnectionStrength(target, targetMode = 'type', connections=None, type="pre", partnerMode="type",connectionType = "dendrites", normalized = True, etc = True, etcTreshhold=0.03, mergePAMSubtypes=False, mergePAMsupertype = False, mergeOthers = True):
    if not isinstance(connections, pd.DataFrame):
        raise ValueError("No connections dataframe passed.")
    if connectionType == "dendrites":
        conn = extractDendritesPerType(target,targetMode=targetMode, connections=connections)
    if connectionType == "axons":
        conn = extractAxonsPerType(target,targetMode=targetMode,connections=connections)
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


def plotPAMStatistic(targets, targetMode = "type",etcTreshhold=0.03, partnerMode="type", connections=None, normalized=True, title="PAM-PAM Connections Statistic", mergePAMSubtypes=False, mergePAMsupertype=False, mergeOthers=True, yLabel="% of synaptic connections", color_dict=PAM_colors,settingsSpec="", ax = None):
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
    """
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


