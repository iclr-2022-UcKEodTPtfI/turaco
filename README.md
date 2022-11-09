# Turaco Language and Experiments

This repository contains the implementation of the Turaco programming language and its analysis, and the experiments in Sections 2 and 5 of the paper "Optimal Data Sampling for Training Neural Surrogates of Programs".

## Install

Requires Python >= 3.9.


First create a virtual environment and install the requirements:
```
python -m venv env
. env/bin/activate
pip install -r requirements.txt
```
Then install Turaco:
```
pip install -e analysis
```

### Renderer

The renderer example also requires the following dependencies:

* g++ (GCC) 11.1.0
* Panda3D 1.10.9
* OpenGL version string: 4.6.0 NVIDIA 465.31

The environment variables `P3D_INCLUDE_PATH` and `P3D_LIB_PATH` must be set to point to the Panda3D library.

## Usage

The `examples` directory has several examples, including the examples from Sections 2 and 5.

To run the standard interpretation of Turaco on a program, run:
```
python -m turaco --program {{PROGRAM_FILE}} interpret --input INPUT_NAME:INPUT_VALUE INPUT_NAME:INPUT_VALUE ...
```

For example, to execute the example in Section 2 of the paper, run:
```
python -m turaco --program examples/daytime.t interpret --input sunPosition:0.5 emission:0.2
```

To run the complexity interpretation of Turaco on a program, run:
```
python -m turaco --program {{PROGRAM_FILE}} complexity --input BETA
```

For example, to calculate the complexity of the daytime example in Section 2 of the paper, run
```
python -m turaco --program examples/daytime.t complexity --input 1.4
```

## Code Layout

The `analysis` directory contains the implementation of Turaco. The `examples` directory contains several examples of Turaco programs. The `example-experiments` directory contains the implementation of the experiments of Section 2. The `renderer-experiments` contains the implementation of the experiments in Section 2.

## Section 2

To run the examples in Section 2:
```
cd example-experiments
python plot.py --program daytime.t:0.4 nighttime.t:0.5 twilight.t:0.1
```

## Section 5

To run the examples in Section 5:
```
cd renderer-experiments
pushd renderer
./build.sh
for s in base_day base_night top_day top_night; do ./run.sh --log --frames 100 --scene $s; done
popd
```
Each scene will print out an identifier like `166XXXXXXXXXXXX`, which we will call `IDENTIFIER`. Copy `textures_IDENTIFIER` to `renderer-experiments/data/textures_IDENTIFIER`. Then:
```
python main.py --identifier IDENTIFIER dataset
```
To construct a stratified surrogate at a given number of data points:
```
python main.py --identifier IDENTIFIER --from 1 --to 6 --steps 20 generate --n NUMBER_OF_SAMPLES --trial 0 --type (optimal/test/uniform)
```
The above command writes a file `data/surrogates/(optimal/test/uniform)_NUMBER_OF_SAMPLES_0.data`, which we will call `SURROGATE_FILE`.
To run this in the renderer:
```
cd renderer-experiments
pushd renderer
/run.sh --surrogate SURROGATE_FILE
```
