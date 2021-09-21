# AliNA
___

(ALIgned Nucleic Acids) is an RNA secondary structures prediction algorithm based on deep learning. It is used to predict the DotBracket notation of secondary structure based on primary sequence.

# Installation
___

- Install Python (version 3.8 or greater)
- Download the repository: 
    ```
    wget https://github.com/Arty40m/AliNA.git
    ```
    Alternatively, it is possible to download the archive directly.

- Create virtual environment and install the required packages:

    - Using pip:
    ```
    python3 -m venv Alina_venv
    . Alina_venv/bin/activate
    pip install -r requirements.txt
    ```
    - Using conda:
    ```
    conda create --name Alina_venv python=3.9
    conda activate Alina_venv
    conda install --file requirements.txt
    ```

In case you use Windows it may also be required to [install](www.microsoft.com) the last version of Visual C++ redistributables (used by TensorFlow).

# Usage
___
```
python3 fold.py [-f, -s] [file, sequence]
```
**-s** is the input nucleotides sequence
**-f** is the path to the .fasta file

#### Optional arguments:
**-o** is the path to the file which will store the prediction results. If this argument is not specified explicitly, the default value will be used:
Prediction_(.fasta file name)

**-th** is the sensitivity threshold when processing predictions **[0, 1]**. The lower the threshold the more bonds will be included into the predicted structure. Unless specified explicitly, the default value is 0.5.

**--help, -h** show additional info about the available commands.

#### Additional information:

- The algorithm will ignore strands that are longer than 256 nucleotides.
- T symbol (thymine) will be automatically replaced with U (uracil).
- All the unknown symbols will be replaced with N (padding).
- Predictions containing exceedingly higher-order knots that can not be processed will be ignored. Usually this could happen in case random symbols were used in the input sequence or if the sensitivity threshold was set too low. 
