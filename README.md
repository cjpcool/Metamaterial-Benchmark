# Metamaterial-Benchmark
Unified representations for metamaterial datasets.

This project aims to collect various metamaterial datasets, and create a unified graph representation for them, benchmarking metamaterial with graph-based methods.

## Lattice description:
  Only consider a unit cell M = $(L, A, P, X, X_L, E)$, 
  * 𝐋 : Lattice structure. 
    Lengths & angles: (a, b, c, alpha, beta, gamma). E.g., cubic: (1,1,1, 90, 90, 90)
    Vector representation $𝐋 \in 𝑹^{(3×3)}$
  * $𝐗_𝐿$: Lattice attribute
    21 independent stiffness constants: 3D anisotropic stiffness tensor characterized by its 21 independent elastic constants.
    Effective mechanical properties. 
    (1. Young's modulus, 2. shear modulus, 3. Poisson's ratio in the global x-,y-,z-direction)
    * Young‘s modulus:  Ex  = 3.34E-03, Ey  = 3.34E-03, Ez  = 3.34E-03
    *  shear modulus:  Gyz = 5.31E-06, Gxz = 5.31E-06, Gxy = 5.31E-06
    * Poisson's ratio: nuyz = 0.000, nuxz = -0., nuxy = 0., nuzy = 0., nuzx = -0., nuyx = 0.
  * $A \in 𝑹^{(𝑵 × 𝑵)}$  or $A \in 𝑹^{(𝟐 × 𝑴)}$: Edge connection, Adj or Edge set; M denotes edge number.
  * $P \in 𝑹^{(𝑵 × 𝟑)}$  : 3D possition of N nodes (truss)
  * $𝐗 \in 𝑹^{(𝑵 × 𝒅_1)}$:  Node attribute. d1 = 2 (node type (cross node))
      Node type: atom element type,  cross node
  * $E \in 𝑹^{(𝑴 ×𝒅_2)}$  : Edge attributes. d2 = 2, (edge type, edge thickness, edge freedom)
    Periodical information: indicates how many unit cell it connecting.
    Edge type: C=N…, nearest node

## Metamaterial Datasets
In updating...
* **LatticeStiffness**: Bastek J H, Kumar S, Telgen B, et al. Inverting the structure–property map of truss metamaterials by deep learning[J]. Proceedings of the National Academy of Sciences, 2022, 119(1): e2111505119.
* **LatticeModulus**: Lumpe T S, Stankovic T. Exploring the property space of periodic cellular structures based on crystal networks[J]. Proceedings of the National Academy of Sciences, 2021, 118(7): e2003504118.

## Data Download
To use dataset LatticeModulus, please unzip LatticeModulus.zip to [PATH\LatticeModulus], and load dataset by running:
~~~python
dataset = LatticeModulus('[your unzip path]\LatticeModulus', file_name='data')
~~~

To use dataset LatticeStiffness, please run:
~~~python
dataset = LatticeStiffness('[your path]\LatticeStiffness', file_name='training')
~~~
The dataset will be downloaded and processed automatically.


* Statistics
    
  | **Properties**  **\Datasets** | **Lattice number** | **Max\Min\Avg node num** | **Max\Min\Avg edge num** | **Lattice properties**                                       | **Edge feat** |
  | ----------------------------- | ------------------ | ------------------------ | ------------------------ | ------------------------------------------------------------ | ------------- |
  | LatticeStiffness                     | 1,048,575          | 50\8\20                  | 118\8\46                 | 21 elastic constants   | Edge Diameter |
  | LatticeModulus                       | 17,222             | 4224\6\91                | 7008\113\6               | Three mechanical properties | None          |
  


* Visualization examples ![image](https://github.com/user-attachments/assets/46fa2912-7e66-4d01-be05-0328e9303bc9)

# Environment preparation
~~~
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pyg -c pyg
conda install pandas
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install e3nn matplotlib scikit-learn plotly ase tensorboard==2.17.0 wandb
~~~