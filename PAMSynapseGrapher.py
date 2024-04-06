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

from utils import getFilteredConnections, plotPAMStatistic 


## setting up Neuprint Client
## using dotenv to import Janelia PAT
from dotenv import load_dotenv
import os
load_dotenv()

## fetching Janelia client
client = neu.Client('https://neuprint.janelia.org/', dataset='hemibrain:v1.2.1', token=os.environ.get("JANELIA_PAT"))



### get PAM-PAM connections
filteredConnections = getFilteredConnections(silent=False)
### retrieving all connections between PAM and non-PAM neurons
filteredPAMConnections = getFilteredConnections("^PAM.*","^.*", minWeight=1)


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