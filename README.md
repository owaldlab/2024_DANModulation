# 2024_DANModulation

This project analyzes the connective properties of PAM-type dopaminergic neurons in *Drosophila Elegans*, using the *Drosophila* Hemibrain connectome dataset.
It features two python scripts with a GUI: PAMConnectionGrapher and PAMSynapsePlotter.

They allow quick, and dynamic visualization of connectomic information across multiple dimensions and employing different filter.
- Connection Grapher:
  - Visualizes a stacked bar chart of incoming and outgoing connections of any PAM type or instance to all other types or instances of PAM or any neuron. It furthermore allows anatomical filtering (mushroom body, non mushroom body). 
- Synapse Plotter:
  - Visualizes 2D projections of synapses to/from any PAM instance or type, coloring them based on connection modality (e.g. same type, same instance, etc.) and anatomical location (mushroom body, non-mushroom body).
  - Allows anatomical filtering.

![Screenshot of GUI tools]("Screenshot 1.png")

---

# Getting Started

Follow these steps to set up and run the Python script from this repository on your local machine.

## Prerequisites

Ensure you have the following installed:
- [Git](https://git-scm.com/)
- [Python 3](https://www.python.org/downloads/)

## Step-by-Step Guide

### 1. Clone the Repository

First, clone the repository to your local machine. Open your terminal (or Git Bash on Windows) and run:

```sh
git clone <your-repo-url>
```

Replace `<your-repo-url>` with the URL of your Git repository.

### 2. Install venv

A virtual environment to manage the project's dependencies. Run the following commands:

```sh
pipx install virtualenv
```

### 3. Navigate to the Project Directory

Change your directory to the newly cloned repository:

```sh
cd <your-repo-directory>
```

Replace `<your-repo-directory>` with the name of your repository folder.

### 4. Activate the Virtual Environment

Activate the virtual environment within the project folder to isolate your project dependencies from other projects.

#### On macOS/Linux:
```sh
source venv/bin/activate
```

#### On Windows:
```sh
.\venv\Scripts\activate
```

### 5. Install Dependencies

With the virtual environment activated, install the required dependencies listed in the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

### 6. Run the Python Script

Now you can run PAMSynapsePlotter.py or PAMConnectionGrapher.py 

```sh
python PAMSynapsePlotter.py
```

## Deactivating the Virtual Environment

Once you are done, deactivate the virtual environment by running:

```sh
deactivate
```

## Additional Information

- If you need to install additional packages, use `pip install <package-name>` and then update `requirements.txt` with `pip freeze > requirements.txt`.
- Make sure to always activate the virtual environment when working on this project to avoid conflicts with other projects' dependencies.

---
