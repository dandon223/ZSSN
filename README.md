# Instalacja

1. Conda do środowiska https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
2. Create environment (prawdopoobnie nie wszystko jest potrzebne ale dziala (zakladam ze cpu wystarczy))
```
conda env create -f environment.yml
```
3. Activate the environment
```
conda activate ZSSN
```
4. Pytorch verification
```
import torch
x = torch.rand(5, 3)
print(x)
```

# Strony z materialami

* https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html?highlight=temporal_signal_split#applications

* https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html

* https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gconv_gru.GConvGRU