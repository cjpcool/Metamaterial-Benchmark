# 🚀 `MetamatBench`: Integrating Heterogeneous Data, Computational Tools, and Visual Interface for Metamaterial Discovery
<p align="center">
    <img src="https://github.com/user-attachments/assets/c8dd5b2b-bc68-41ce-bd97-53051b63a191" alt="MetaBench" width="80%">
</p>

---

## 👥 Authors and Affiliations

**Jianpeng Chen¹**, **Wangzhi Zhan¹**, **Haohui Wang¹**, **Zian Jia²⁵**, **Jingru Gan³**, **Junkai Zhang³**, **Jingyuan Qi¹**, **Tingwei Chen¹**, **Lifu Huang⁴**, **Muhao Chen⁴**, **Ling Li⁵**, **Wei Wang³**, **Dawei Zhou¹**

¹ Virginia Tech, ² Princeton University, ³ University of California, Los Angeles, ⁴ University of California, Davis, ⁵ University of Pennsylvania





## 📌 Abstract
Metamaterials—engineered materials with architected structures across multiple length scales—offer unprecedented and tunable mechanical properties that surpass conventional materials. However, leveraging **machine learning (ML) for metamaterial discovery** is hindered by three fundamental challenges:

1. **(C1) Data Heterogeneity Challenge**: Variations in data sources, composition scales, and structure categories.
2. **(C2) Model Complexity Challenge**: Geometric constraints of ML models complicate adaptation to metamaterial structures.
3. **(C3) Human-AI Collaboration Challenge**: The dual black-box nature of sophisticated ML models and the need for intuitive user interfaces.

### 🔹 `MetamatBench` - A Unified Solution
We introduce **MetamatBench**, a unified framework that addresses these challenges on three levels:

- **📊 Data Level**: Standardizes and integrates **5 heterogeneous, multi-modal metamaterial datasets**.
- **🧠 ML Level**: Provides a toolkit of **17 state-of-the-art ML methods**, a **comprehensive evaluation suite** with **12 performance metrics**, and **finite element-based assessments** for accurate model validation.
- **👨‍💻 User Level**: Features an **interactive visual interface** to bridge the gap between complex ML techniques and non-ML researchers.

🔗 **Visual-Interactive Interface**: [MetamatBench Online](http://zhoulab-1.cs.vt.edu:5550)  
🔗 **Codebase**: [GitHub Repository](https://github.com/cjpcool/Metamaterial-Benchmark)

---

## 📐 Unified Representation
<p align="center">
    <img src="https://github.com/user-attachments/assets/194a3fba-bfa4-4798-91f0-5eb64db3c272" alt="Unified Representation" width="40%">
</p>

To address **data heterogeneity (C1)**, we introduce a **unified metamaterial representation** for 3D graph datasets, enabling benchmarking with advanced geometric graph learning methods. This representation considers a **six-dimensional learning space**, emphasizing **structural lattice characteristics** over atomic composition.

In general, a metamaterial is represented as:
\[
\mathcal{M}(\mathbf{L}, \mathcal{U}, \mathbf{y})
\]
where:

- **Metamaterial Property (D1)**: $\mathbf{y} \in \mathbb{R}^{d}$ represents mechanical properties.
- **Lattice Representation (D2)**: $\mathbf{L} = [\mathbf{l}_0, \mathbf{l}_1, \mathbf{l}_2]^\mathrm{T} \in \mathbb{R}^{3 \times 3}$ captures periodic angles and lattice lengths.
- **Unit Cell Representation (D3-D6)**:
  - **Node Coordinates (D3)**: 3D Cartesian system positions.
  - **Node Attributes (D4)**: Encodes four node types (face, corner, edge, inner).
  - **Edge Connections (D5)**: Defines graph edges.
  - **Edge Attributes (D6)**: Stores auxiliary properties like edge diameter.

---

## Models Toolbox
*The statistics of comparison methods. `*` indicates conditional generation support.*  
Abbreviations:  
- **Trans Inv.** (Translation Invariance), **Glob** (Global), **Equiv.** (Equivariant), **Rot.** (Rotation),  
- **VAE** (Variational Auto Encoder), **Diff** (Diffusion), **MPNN** (Message Passing Neural Networks), **GCN** (Graph Convolutional Networks),  
- **LatDiff** (Latent Diffusion), **Perm.** (Permutation).  

| **Method** | **Task** | **Target Graph** | **Periodicity** | **Symmetry** | **Backbone** | **Publication** |
|------------|----------|------------------|-----------------|--------------|--------------|-----------------|
| **Generation Methods** | | | | | | |
| [CDVAE](https://arxiv.org/abs/2110.06197) | Generation | Crystal | Trans Inv. | Inv Enc + Equiv Dec | VAE+Diff | ICLR'22 |
| [Cond-CDVAE](https://www.nature.com/articles/s41524-024-01443-y) | Generation | Crystal | Trans Inv. | Inv Enc + Equiv Dec | VAE+Diff | Nat. Commun.'24 |
| [SyMat](https://arxiv.org/abs/2307.02707) | Generation | Crystal | Trans Inv. | Inv Enc + Inv Dec | VAE+Diff | NeurIPS'23 |
| [DiffCSP](https://arxiv.org/abs/2309.04475) | Generation | Crystal | Trans Inv. | Equiv. (lacks lattice perm. eq.) | Diff | NeurIPS'23 |
| [EquiCSP](https://openreview.net/forum?id=VRv8KjJNuj) | Generation | Crystal | Trans Inv. + Perm Eq. | Equiv. | Diff | ICML'24 |
| [Crystal-Text-LLM](https://openreview.net/forum?id=0r5DE2ZSwJ) | Generation | Crystal | N/A | N/A | LLaMA-2 | AI4Mat'23 |
| [CrystaLLM](https://www.nature.com/articles/s41467-024-54639-7) | Generation | Crystal | N/A | N/A | LLaMA-2 | Nat. Commun.'24 |
| [EDM](https://proceedings.mlr.press/v162/hoogeboom22a) | Generation | Molecule | N/A | Equiv. | Diff | PMLR'22 |
| [GeoLDM](https://proceedings.mlr.press/v202/xu23n.html) | Generation | Molecule | N/A | Equiv. | LatDiff | PMLR'23 |
| **Prediction Methods** | | | | | | |
| [SchNet](https://arxiv.org/abs/1706.08566) | Prediction | Molecule | N/A | Glob Inv | MPNN | NeurIPS'17 |
| [SphereNet](https://arxiv.org/abs/2102.05013) | Prediction | Molecule | N/A | Glob Inv | MPNN | ICLR'22 |
| [ViSNet](https://www.nature.com/articles/s41467-023-43720-2) | Prediction | Molecule | Trans + Rot Inv. | Equiv. | MPNN | Nat. Commun.'23 |
| [Equiformer](https://openreview.net/forum?id=KwmPfARgOTD) | Prediction | Molecule | N/A | Equiv. | MPNN | ICLR'23 |
| [ALIGNN](https://www.nature.com/articles/s41524-021-00650-1) | Prediction | Crystal | Trans Inv. | Glob Inv | GNN | npj Comput. Mater.'21 |
| [CGCNN](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) | Prediction | Crystal | Trans Inv. | Glob Inv. | GCN | PRL'18 |
| [UniTruss](https://www.nature.com/articles/s41467-023-42068-x) | Prediction | Metamaterial | N/A | N/A | VAE | Nat. Commun.'23 |
| [MACE+ve](https://openreview.net/forum?id=smy4DsUbBo) | Prediction | Metamaterial | N/A | Equiv. | GNN | ICLR'24 |

Overall, to benchmark 3D graph models on metamaterials, the proposed taxonomy in Figure~\ref{fig:overview} covers: 
1. Generative models and predictive models,  
2. Crystal, molecular, and metamaterial graph models,  
3. Models with different periodicity constraints,  
4. Models with symmetry constraints (e.g., equivariant and invariant models), and  
5. Models with various backbones.  

The detailed model taxonomy is summarized in Table~\ref{tab:comparisonModel}.  
Based on this taxonomy, we benchmark two fundamental tasks—prediction and generation—to measure the effectiveness of ML models for metamaterial learning in various application scenarios.

---

## Evaluation Toolbox

<p align="center">
    <img src="https://github.com/user-attachments/assets/3141a753-96f2-4dbc-ab1a-2df74824cda0" alt="evaluationtookbox" width="50%">
</p>

We develop a novel evaluation framework in the ML toolbox to assess metamaterial models. Adopting a multi-perspective approach, our framework provides robust and unbiased evaluations.

---

## 📂 Datasets
### 🔍 **Available Datasets**
| **Dataset**         | **Description**                                           | **Periodic** | **Property**             | **# Sample** | **Source**                                                                       |
|---------------------|-----------------------------------------------------------|--------------|--------------------------|--------------|----------------------------------------------------------------------------------------|
| **MetaModulus**     | Architected truss metamaterials for modulus design        | Yes          | Mechanical properties    | 16,707       | [Link](https://doi.org/10.3929/ethz-b-000457598)                                        |
| **MetaStiffness**   | Architected truss metamaterials for stiffness optimization  | Yes          | Elastic constants        | 1,048,575    | [Link](https://doi.org/10.3929/ethz-b-000520254)                                        |
| **PointCloud**      | Truss metamaterials (3D point cloud representation)         | Yes          | Mechanical properties    | 29,400       | [Link](https://github.com/lychan110/metaset/tree/master/3D_diverse)                      |
| **MetaTruss**       | Architected truss metamaterials for homogeneous stiffness   | Yes          | Homogeneous stiffness    | 965,736      | [Link](https://doi.org/10.3929/ethz-b-000618078)                                        |
| **LagrangianFrame** | Shell and truss metamaterials (2D Lagrangian frame)          | Yes          | Stress and strain        | 53,007       | [Link](https://doi.org/10.3929/ethz-b-000629716)                                        |

---

## 🏗️ Results

### **Generation Evaluation**
*DR: Dangling Restriction, Conn: Connectivity, Sym: Symmetry, Peri: Periodicity, CR: Coverage Recall, CP: Coverage Precision, MET: Mean Evaluation Time (generation time per sample), MTT: Mean Training Time.*

| **Approach** | **Validity ↑** |               |               |               | **Diversity ↑**             |               |               |               | **Cond. Effectiveness ↓** | **Efficiency**    |
|--------------|----------------|---------------|---------------|---------------|-----------------------------|---------------|---------------|---------------|---------------------------|-------------------|
|              | DR (%) | Conn (%) | Sym (%) | Peri (%) | Mean (%) | Cov R. (%) | Cov P. (%) | Mean (%) |  | MET (s) | MTT (ms) |
| **Molecule Targeted Methods** |  |  |  |  |  |  |  |  |  |  |  |
| [EDM](https://proceedings.mlr.press/v162/hoogeboom22a)     | N/A   | N/A   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 982.13   | 3.18   | 161.39   |
| [GeoLDM](https://proceedings.mlr.press/v202/xu23n.html)     | N/A   | N/A   | 0.04   | 0.00   | 0.02   | 0.00   | 0.00   | 0.00   | 60.59    | 2.84   | 606.80   |
| **Crystal Targeted Methods**         |      |       |        |        |       |       |       |       |           |           |
| [CDVAE](https://arxiv.org/abs/2110.06197)                   | N/A   | N/A   | 57.03  | 0.40   | 28.72  | 55.85  | 95.80  | 75.83  | N/A       | 93.00    | 97.42    |
| [DiffCSP](https://arxiv.org/abs/2309.04475)                 | N/A   | N/A   | 34.46  | 6.50   | 20.48  | 95.80  | 96.65  | 96.23  | N/A       | 2.97     | 63.79    |
| [EquiCSP](https://openreview.net/forum?id=VRv8KjJNuj)       | N/A   | N/A   | 55.37  | 3.55   | 29.46  | 100.00 | 52.35  | 76.18  | N/A       | 1.90     | 64.57    |
| [Cond-CDVAE](https://www.nature.com/articles/s41524-024-01443-y) | N/A   | N/A   | 19.37  | 2.00   | 10.69  | 68.60  | 80.50  | 74.55  | 0.2050    | 225.01   | 314.51   |
| [SyMat](https://arxiv.org/abs/2307.02707)                   | N/A   | N/A   | 41.10  | 0.00   | 20.55  | 79.34  | 38.90  | 59.12  | N/A       | 89.49    | 141.20   |
| [CrystaLLM](https://www.nature.com/articles/s41467-024-54639-7) | 3.60  | 26.90  | 76.43  | 92.10  | 49.76  | 100.00 | 100.00 | 100.00 | 0.0983   | 2.08     | 14.51s   |
| [Crystal-Text-LLM](https://openreview.net/forum?id=0r5DE2ZSwJ) | 23.50 | 68.50  | 89.37  | 96.10  | 69.37  | 100.00 | 100.00 | 100.00 | 0.0916   | 46.49    | 708.00   |

---

### **Prediction Evaluation**
*MAE: Mean Absolute Error, R²: R-squared, NRMSE: Normalized Root Mean Square Error, MET: Mean Evaluation Time, MTT: Mean Training Time.*

| **Approach**  | **Young's Modulus**         | **Shear Modulus**         | **Poisson's Ratio**        | **Efficiency (ms)** |
|---------------|-----------------------------|---------------------------|----------------------------|---------------------|
|               | MAE ↓  | R² ↑  | NRMSE ↓ | MAE ↓  | R² ↑  | NRMSE ↓ | MAE ↓  | R² ↑  | NRMSE ↓ | MET  | MTT  |
| **Molecule Targeted Methods** |         |         |         |         |         |         |         |         |         |      |      |
| [SchNet](https://arxiv.org/abs/1706.08566)    | 0.0005704 | 0.2304 | 0.1436 | 0.0001343 | 0.01441 | 0.3958 | 0.3962 | -0.1025 | 0.02029 | 3.34 | 12.39 |
| [SphereNet](https://arxiv.org/abs/2102.05013)   | 0.0004744 | 0.4548 | 0.1185 | 0.0001039 | 0.2839 | 0.08682 | 0.3561 | -0.01795 | 0.02499 | 11.62 | 33.08 |
| [Equiformer](https://openreview.net/forum?id=KwmPfARgOTD) | 0.0006669 | -0.3892 | 0.2469 | 0.0002226 | -1.1149 | 0.3459 | 0.3673 | -0.03171 | 0.05805 | 63.17 | 204.70 |
| [ViSNet](https://www.nature.com/articles/s41467-023-43720-2)    | 0.0006223 | 0.05871 | 0.1506 | 0.06375   | 0.06375 | 0.1003 | 0.3699 | -0.04497 | 0.01916 | 15.63 | 32.33 |
| **Crystal Targeted Methods**  |         |         |         |         |         |         |         |         |         |      |      |
| [CGCNN](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) | 0.0006179 | 0.3550 | 0.1785 | 0.0001475 | 0.09353 | 0.1720 | 0.3922 | 0.04905 | 0.08282 | 10.76 | 48.51 |
| [ALIGNN](https://www.nature.com/articles/s41524-021-00650-1)    | 0.0008320 | -1.2955 | 0.1544 | 0.0001460 | -0.01672 | 0.09749 | 0.31267 | -0.13298 | 0.01634 | 990.34 | 44993.19 |
| **Metamaterial Targeted Methods** |     |         |         |         |         |         |         |         |         |      |      |
| [uniTruss](https://www.nature.com/articles/s41467-023-42068-x)  | 0.0006266 | 0.1812 | 0.1389 | 0.0001451 | 0.06374 | 0.09955 | 0.3970 | -0.1016 | 0.02014 | 1.18  | 0.22  |
| [MACE+ve](https://openreview.net/forum?id=smy4DsUbBo)         | 0.0003882 | 0.6692 | 0.08797| 0.0001211 | 0.1932 | 0.08913 | 0.2881 | 0.1585  | 0.01738 | 27.34 | 304.2 |

---

## 🎨 Visualization Examples
More visualizations please link to 

🔗 **Visual-Interactive Interface**: [MetamatBench Online](http://zhoulab-1.cs.vt.edu:5550)  
<p align="center"> <img src="https://github.com/user-attachments/assets/46fa2912-7e66-4d01-be05-0328e9303bc9" alt="Visualization Examples" width="50%"> </p>


## ⚙️ Environment Setup
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pyg -c pyg
conda install pandas
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install e3nn matplotlib scikit-learn plotly ase tensorboard==2.17.0 wandb
```
**For GeoML** 
```
pip install imageio rdkit
```
**For OCP**
```
cd ocp
pip install -e .
pip install lmdb
```
**For CDVAE**
```
conda install pytorch_lightning
pip install hydra-core-1.3.2 omegaconf-2.3.0 hydra-joblib-launcher python-dotenv-1.0.1 pymatgen p_tqdm
```


##  🚀 Running Examples
### 📥 Datasets Loading
🔹 **LatticeModulus**
```python
dataset = LatticeModulus('[your unzip path]/LatticeModulus', file_name='data')
```
🔹 **LatticeStiffness**
```python
dataset = LatticeStiffness('[your path]/LatticeStiffness', file_name='training')
```
Note: The dataset will be automatically downloaded and processed.
### 🏗️ Model Toolbox Usage
```python
# Step 0: Carefully edit the config file at: configs/[Model]/[Dataset]_config.yml

# Step 1: Initialize model
model = MaceVeModel(model_name='mace_ve', 
                     dataset_name='LatticeModulus', 
                     device=device, root_path='./')

# Step 2: Load dataset
model.load_data()

# Step 3: Load pre-trained model
model.load_model()

# Step 4: Train the model
model.train()

# Step 5: Evaluate the model
r2, nrmse, mae = model.test()
```

### 📐 Evaluation Toolbox Usage
🔹 **Prediction Evaluation**
```python
import numpy as np
from evaluation.prediction_eval import calculate_metrics
# Test data:
pred = np.array([[2.5, 3.5], [4.5, 1.5]])
target = np.array([[3, 4], [5, 2]])

# Use 
r2, nrmse, mae = calculate_metrics(pred, target)
print(f"R^2: {r2}, NRMSE: {nrmse}, MAE: {mae}")
```
🔹 **Generation Evaluation**
1. Please save the generated lattice as follows. Each lattice is saved as a ".npz" file.
```
Step 1: Saving lattices:
            np.savez(lattice_name,
                atom_types=gen_atom_types_list[i],
                lengths=gen_lengths_list[i],
                angles=gen_angles_list[i],
                frac_coords=gen_frac_coords_list[i],
                edge_index=edge_index_list[i],
                prop_list=prop_list[i]
                )
- `cart_coords`: None or cartesian coordinates of each atom, shape `(num_evals, N, 3)`
- `frac_coords`: fractional coordinates of each atom, shape `(num_evals, N, 3)`
- `atom_types`: atomic number of each atom, shape `(num_evals, N)`
- `lengths`: the lengths of the lattice, shape `(num_evals, M, 3)`
- `angles`: the angles of the lattice, shape `(num_evals, M, 3)`
- `num_atoms`: the number of atoms in each material, shape `(num_evals, M)`
- 'prop_list': (12)
```
2. start to evaluate
```python
from evaluation.generation_eval import LatticeEvaluator
from datasets.dataset_truss import LatticeStiffness, LatticeModulus

# Step 2: Construct test data (This step can be omitted if you only evaluate validity.):
dataset = LatticeModulus('[data_root_path]\\LatticeModulus',file_name='data')
split_dict = dataset.get_idx_split(len(dataset), 8000, 2000, seed=42)
test_data = dataset[split_dict['test']]

# Step 3: Instantiate a LatticeEvaluator object, where eval_file_path is the path that the ".npz" files are saved in 1.  
evaluator = LatticeEvaluator(test_datset=test_data[:1000], eval_file_path='[project_path]\gen_results\\[model_name]\\save_results')
## You can also instantiate LatticeEvaluator object via input list of lattices instead of eval_file_path:
# evaluator = LatticeEvaluator(
#         None, None,
#         cart_coords,
#         frac_coords, # can be None
#         node_types,
#         edges, # can be None
#         lattice_vectors) 

# Step 4: evaluate lattices:
# Evaluate diversity and Validity.
evaluator.evaluate_all_uncondition_generation()
## Alternatively, you can also call:
periodicity_ratio, mean_symmetry, connectivity_ratio, dangling_node_ratio = evaluator.eval_graph_validity()
cov_r, cov_p = evaluator.eval_diversity()

# Evaluate conditional effectiveness.
effectiveness = evaluator.eval_condition_effectiveness()
```





## 👏 Contributors

Thanks to all the wonderful contributors who made this project possible! 💡

| [@Junkai Zhang](https://github.com/Jun-Kai-Zhang) | [@Jingru Gan](https://github.com/JingruG) | [@Jingyuan Qi](https://github.com/jingyq1) | [@Wangzhi Zhan](https://github.com/wzhan24) | [@Tingwei Chen](https://github.com/lajictw) |
|:------------------------------------------------:|:-----------------------------------------:|:-----------------------------------------:|:-----------------------------------------:|:-----------------------------------------:|
| ![Junkai Zhang](https://avatars.githubusercontent.com/u/Jun-Kai-Zhang?v=4) | ![Jingru Gan](https://avatars.githubusercontent.com/u/JingruG?v=4) | ![Jingyuan Qi](https://avatars.githubusercontent.com/u/jingyq1?v=4) | ![Wangzhi Zhan](https://avatars.githubusercontent.com/u/wzhan24?v=4) | ![Tingwei Chen](https://avatars.githubusercontent.com/u/lajictw?v=4) |

🎉 *Your contributions are always welcome!*



