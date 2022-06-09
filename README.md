AliNA
===

(ALIgned Nucleic Acids) is an RNA secondary structures prediction algorithm based on deep learning. It is used to predict the DotBracket notation of secondary structure based on primary sequence.
![Title](/imgs/alina.png)

Installation
===

- Install Python (version 3.8 or greater)
- Download the repository and cd there: 
    ```
    git clone https://github.com/Arty40m/AliNA.git
	cd AliNA
    ```
    Tensorflow 2.5.0 is required for correct working of AliNA. We recommend to use pip to install this particular version of tensorflow. But you can do that within conda environment as well.

- Create virtual environment:

    - Using pip:
    ```
    python3 -m venv Alina_venv
    . Alina_venv/bin/activate
    ```
    - Using conda:
    ```
    conda create --name Alina_venv python=3.9
    conda activate Alina_venv
    ```
- Install the required packages
    ```
    python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt
    ```

In case you use Windows it may also be required to [install](https://www.microsoft.com) the last version of Visual C++ redistributables (used by TensorFlow).

Usage
===

__Quick example:__

```
- RNA sequence
    python3 fold.py -i GGAGCAUUACCCCCCCAAAACCCUGGGGAUACAGGGCCCA
- Fasta file
    python3 fold.py -m file -i path_to_file.fasta -o path_to_predictions_file.out
```

```
python3 fold.py [-m {seq,file}] -i <sequence OR fasta_file> [-o output_file] [-th threshold]
```
**-m**  - prediction mode: "seq" - for single sequence typed in arguments. "file" - for multiple predictions from .fasta file

**-i** - input sequence or path to fasta file

**-o** - path to the output file for "file" mode. Default - Prediction_<input file name>

**-th** to specify the sensitivity threshold when processing predictions **[0, 1]**. The lower the threshold the more bonds will be included into the predicted structure. Unless specified explicitly, the default value is 0.5.

**---help, -h** show additional info about the available commands.

#### Additional information:

- The algorithm will ignore strands that are longer than 256 nucleotides.
- T symbol (thymine) will be automatically replaced with U (uracil).
- All the unknown symbols will be replaced with N (padding).
- Predictions containing exceedingly higher-order knots that can not be processed will be ignored. Usually this could happen in case random symbols were used in the input sequence or if the sensitivity threshold was set too low. 
