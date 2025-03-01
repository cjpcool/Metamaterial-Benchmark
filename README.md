# `MatamatBench`: Integrating Heterogeneous Data, Computational Tools, and Visual Interface for Metamaterial Discovery
![MetaBench](https://github.com/user-attachments/assets/c8dd5b2b-bc68-41ce-bd97-53051b63a191)


## Abstract
Metamaterials, engineered materials with architected structures across multiple length scales, offer unprecedented and tunable mechanical properties that surpass those of conventional materials. However, leveraging advanced machine learning (ML) for metamaterial discovery is hindered by three fundamental challenges: {(C1) Data Heterogeneity Challenge} arises from heterogeneous data sources, heterogeneous composition scales, and heterogeneous structure categories; {(C2) Model Complexity Challenge} stems from the intricate geometric constraints of ML models, which complicate their adaptation to metamaterial structures; and {(C3) Human-AI Collaboration Challenge} comes from the ``dual black-box'' nature of sophisticated ML models and the need for intuitive user interfaces. 
To tackle these challenges, we introduce a unified framework, named `MetamatBench`, that operates on three levels. 
(1) At the \emph{data level}, we integrate and standardize 5 heterogeneous, multi-modal metamaterial datasets.
(2) The \emph{ML level} provides a comprehensive toolkit that adapts 17 state-of-the-art ML methods for metamaterial discovery. It also includes a comprehensive evaluation suite with 12 novel performance metrics with finite element-based assessments to ensure accurate and reliable model validation.
(3) The \emph{user level} features a visual-interactive interface that bridges the gap between complex ML techniques and non-ML researchers, advancing property prediction and inverse design of metamaterials for research and applications.
`MetamatBench` offers a unified platform deployed at [interfalceurl](http://zhoulab-1.cs.vt.edu:5550) that enables machine learning researchers and practitioners to develop and evaluate new methodologies in metamaterial discovery. 
For accessibility and reproducibility, we open-source our benchmark and the codebase at [codebaseurl](https://github.com/cjpcool/Metamaterial-Benchmark).


## Unified Representation:
  Only consider a metamaterial representation $\mathcal{M}(\mathbf{L}, \mathcal{U}, \mathbf{y})$, 
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

# geolDM
pip install imageio rdkit
~~~
~~~
cd ocp
pip install -e .
pip insall lmdb
~~~

For CDVAE:
~~~
conda install pytorch_lightning
pip install hydra-core-1.3.2 omegaconf-2.3.0 hydra-joblib-launcher python-dotenv-1.0.1 pymatgen p_tqdm
~~~