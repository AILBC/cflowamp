# CFlowAMP

## Project Description

CFlowAMP is a protein sequence generation project based on Flow Matching and ESM (Evolutionary Scale Modeling). The project uses Transformer architecture and flow matching techniques to generate protein sequences with specific properties.

## Installation

1. Clone or download the project locally.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Generating Samples

Run `inference.py` to generate protein sequence samples:

```bash
python inference.py
```

This script will use pre-trained models to generate a specified number of sequences and save them to the `esmflow/gen_out/` directory.

### Training Models

- `flow_trainer.py`: Train the flow matching model
- `esm_gen.py`: AMP generation related code

## Data

- `data/pos_data.csv`: Positive sample data
- `data/neg_data.csv`: Negative sample data

Data contains protein sequences and their properties.

## Models

Pre-trained models are stored in the `model/` directory:

- `autoencoder.pt`: Dimensionality reduction model
- `flow_model86.pt`: Flow matching model
- `calibration_model.pt`: Calibration model

## Dependencies

Main dependencies include PyTorch, NumPy, Pandas, Scikit-learn, etc. See `requirements.txt` for details.
