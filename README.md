## Setup Guide for Multilingual Bot with Ollama

Follow the steps below to set up the Multilingual Bot project using Ollama.

### Step 1: Download and Install Python
Download Python from the official website:
[Download Python](https://www.anaconda.com/download)

Install Python and ensure it is added to your system's environment variables.

### Step 2: Download and Install Anaconda

Download Anaconda: 

[Download Anaconda](https://anaconda.org/)

After installation, add the following paths to your environment variables:

anaconda3/Scripts

anaconda3/Library/bin

### Step 3: Download and Install Ollama

Download Ollama for Windows: 

[Download Ollama](https://ollama.com/download/windows)

Add the Ollama installation folder to your system’s environment variables.

### Step 4: Clone the Multilingual_Bot_Ollama Repository

Clone the repository to your local machine using Visual Studio Code (VS Code) or your preferred code editor.
bash

git clone <repository-url>  (Use this command to clone)

### Step 5: Set Up the Conda Environment

Open a command prompt inside the project folder.

Create a new Conda environment:


conda create --name env

Activate the environment:


conda activate env

Note: Ensure all subsequent commands are run within this environment.

### Step 6: Install PyTorch and CUDA

Run the following command to install PyTorch, TorchVision, and CUDA (adjust the CUDA version based on your system’s compatibility):


conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

### Step 7: Install Python Dependencies

Install all required dependencies from the requirements.txt file:


pip install -r requirements.txt


### Step 8: Set Up Ollama Model and Server

In a separate terminal, pull the Ollama model:


ollama pull llama3.1:8b

Once the model is successfully downloaded, start the Ollama server:


ollama serve

### Step 9: Start the FastAPI Server

Return to your project directory and run the FastAPI server:


fastapi run app.py
