install numpy version 1.24.1 if getting "RuntimeError: Numpy is not available"
`pip install numpy==1.24.1`
NO: latest version of numpy (1.24.1) has np.int deprecated so that doesn't work
instead use `pip install "numpy<1.24.0"` which installs version `1.23.5`
I initially had version `1.22.1`
difference between generating datasets and reading from saved files. Code needs to be
changed back and forth for each parameter change.

same thing for predictions and embeddings data
    filepath also had to be changed

change `torch.save` to `learn.save` when saving and loading the model
