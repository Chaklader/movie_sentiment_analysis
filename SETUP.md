# Make sure your rnn environment is activated
conda activate rnn

# Install ipykernel
pip install ipykernel

# Register the kernel with Jupyter
python -m ipykernel install --user --name rnn --display-name "Python (rnn)"