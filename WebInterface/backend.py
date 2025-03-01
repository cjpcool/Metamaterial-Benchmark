import os.path
import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(root_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import streamlit as st
from datasets import LatticeModulus, LatticeStiffness
from visualization.homo3D import homo3D
from visualization.vis import visualizeLattice_interactive, visualizeLattice, plot_ellipsoid_colormap_modulus, \
    plot_directional_modulus_from_poissons
from visualization.voxel import generate_voxel, visualizeVox
from visualization.visualize_Cij import visualizeCij
from visualization.visualize_rankboard import plot_rankingboard
from wrappers import MaceVeModel

import shutil

import os
import torch


class ResearchBackend:
    def __init__(self):
        self.datasets = ["MetaModulus"]
        self.tasks=['Prediction', 'Generation']
        self.prediction_results = pd.read_csv(f'{root_path}/data/comparison/MetaModulus_Prediction.csv')
        self.generation_results = pd.read_csv(f'{root_path}/data/comparison/MetaModulus_Generation.csv')
        self.rank_board_img_root = f'{root_path}/data/images'
        # self.metrics = ["Young_R2", "Young_MAE", "Young_NRMSE"]
        # self.metrics
        self.dataset_df = None

        # self.dataset_root= 'D:\\项目\\Material design\\code_data\\data'
        self.dataset_root = '/home/jianpengc/datasets/metamaterial'

        self.dataset_visualization_root = f'{root_path}/dataset_results'
        self.dataset_visualization_path = os.path.join(self.dataset_visualization_root, 'visualization.png')
        self.dataset_simulation_path = os.path.join(self.dataset_visualization_root, 'simulation.png')
        self.dataset_voxel_path = os.path.join(self.dataset_visualization_root, 'voxel.png')

        self.default_model_interaction_root = f'{root_path}/model_results'
        self.default_model_interaction_path = os.path.join(self.default_model_interaction_root, 'visualization.png')
        self.default_prediction_path = os.path.join(self.default_model_interaction_root, 'prediction.png')

        DEVICE_ID = '3'
        # os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
        self.device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')


    def load_dataset_stats(self, csv_path):
        """Load dataset statistics from CSV file"""
        self.dataset_df = pd.read_csv(csv_path)
        return self.dataset_df
    
    def get_image_path(self, dataset, task, metric):
        """Construct image path based on selected options"""
        self.prediction_results = pd.read_csv(f'{root_path}/data/comparison/{dataset}_Prediction.csv')
        self.generation_results = pd.read_csv(f'{root_path}/data/comparison/{dataset}_Generation.csv')
        file_name = os.path.join(f"{self.rank_board_img_root}/{dataset}_{task}_{metric}.png")
        print(file_name)
        print(file_name)
        if not os.path.exists(file_name):
            if task == 'Prediction':
                df = self.prediction_results
            elif task == 'Generation':
                df = self.generation_results
            plot_rankingboard(df, dataset, task, metric, save_path=self.rank_board_img_root)
        return file_name

    def get_metric_description(self, dataset, metric):
        """Generate content for metric description (to be implemented later)"""
        # Placeholder implementation
        return f"Description for {dataset} using {metric}\nLearn more at [Datasets](https://example.com)"
    
    def get_dataset_examples(self, dataset_name):
        """Get example data points for selected dataset (to be implemented)"""
        # Placeholder implementation
        return [f"Example1_{dataset_name}", f"Example2_{dataset_name}"]
    
    def get_visualization_images(self, data_point):
        """Get visualization image paths for selected data point (to be implemented)"""
        # Placeholder implementation
        return {
            "left": "data/images/left_example.png",
            "right": "data/images/right_example.png"
        }

    def clear_dataset_results(self):
        for root, dirs, files in os.walk(self.dataset_visualization_root):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)

    def dataset_info_visualize(self, index, datasetname):
        # self.clear_dataset_results()
        if 'Meta' in datasetname:
            if datasetname == "MetaModulus":
                dataset = LatticeModulus(os.path.join(self.dataset_root,'LatticeModulus'), file_name='data')
            elif datasetname == "MetaStiffness":
                dataset = LatticeStiffness(os.path.join(self.dataset_root, 'LatticeStiffness'), file_name='training')
            data = dataset[int(index)]
            cart_coords = data.cart_coords.numpy()
            edge_index = data.edge_index.numpy()

            self.dataset_visualization_path = os.path.join(self.dataset_visualization_root, 'visualization.png')
            # visualizeLattice_interactive(cart_coords, edge_index, file_name=self.dataset_visualization_path)
            visualizeLattice(cart_coords, edge_index, save_dir=self.dataset_visualization_path, dpi=150)

        elif datasetname == 'PointCloud':
            self.dataset_visualization_path = os.path.join(self.dataset_visualization_root, 'visualization.png')
            shutil.copy(f'{root_path}/data/dataset_info/Cloud_point_samples.png', self.dataset_visualization_path)

        return self.dataset_visualization_path

    def interaction_data_visualize(self, index, datasetname):
        # self.clear_dataset_results()
        if 'Meta' in datasetname:
            if datasetname == "MetaModulus":
                dataset = LatticeModulus(os.path.join(self.dataset_root, 'LatticeModulus'), file_name='data')
            elif datasetname == "MetaStiffness":
                dataset = LatticeStiffness(os.path.join(self.dataset_root, 'LatticeStiffness'), file_name='training')
            data = dataset[int(index)]
            cart_coords = data.cart_coords.numpy()
            edge_index = data.edge_index.numpy()

            self.default_model_interaction_path = os.path.join(self.default_model_interaction_root, 'visualization.png')
            # visualizeLattice_interactive(cart_coords, edge_index, file_name=self.dataset_visualization_path)
            visualizeLattice(cart_coords, edge_index, save_dir=self.default_model_interaction_path, dpi=150)

        return self.default_model_interaction_path

    def dataset_info_vox(self, index, vox_size, datasetname):
        vox_size = int(vox_size)
        if datasetname == "MetaModulus":
            dataset = LatticeModulus(os.path.join(self.dataset_root, 'LatticeModulus'), file_name='data')
        elif datasetname == 'MetaStiffness':
            dataset = LatticeStiffness(os.path.join(self.dataset_root, 'LatticeStiffness'), file_name='training')

        self.dataset_voxel_path = os.path.join(self.dataset_visualization_root, 'voxel.png')
        self.dataset_simulation_path = os.path.join(self.dataset_visualization_root, 'simulation.png')
        data = dataset[int(index)]
        cart_coords = data.cart_coords.numpy()
        edge_index = data.edge_index.numpy()
        voxel, Density = generate_voxel(vox_size, cart_coords, edge_index, radius=0.1)
        visualizeVox(voxel, self.dataset_voxel_path)

        return voxel, self.dataset_voxel_path


    def dataset_info_simulation(self, voxel):
        CH = homo3D(1, 1, 1, 0.5769, 0.3846, voxel)
        plt = visualizeCij(CH, 50)
        plt.savefig(self.dataset_simulation_path,bbox_inches='tight')
        plt.close()
        return self.dataset_simulation_path

    def dataset_info_vox_and_simulation(self, index, vox_size, datasetname):
        vox_size = int(vox_size)
        if datasetname == "MetaModulus":
            dataset = LatticeModulus(os.path.join(self.dataset_root,'LatticeModulus'), file_name='data')
        elif datasetname == 'MetaStiffness':
            dataset = LatticeStiffness(os.path.join(self.dataset_root,'LatticeStiffness'), file_name='training')

        self.dataset_voxel_path = os.path.join(self.dataset_visualization_root, 'voxel.png')
        self.dataset_simulation_path = os.path.join(self.dataset_visualization_root, 'simulation.png')
        data = dataset[int(index)]
        cart_coords = data.cart_coords.numpy()
        edge_index = data.edge_index.numpy()
        voxel, Density = generate_voxel(vox_size, cart_coords, edge_index, radius=0.1)
        visualizeVox(voxel, self.dataset_voxel_path)
        CH = homo3D(1, 1, 1, 0.5769, 0.3846, voxel)
        plt = visualizeCij(CH, 50)
        plt.savefig(self.dataset_simulation_path)
        plt.close()
        return self.dataset_voxel_path, self.dataset_simulation_path

    def method_generation(self):
        pass
    def method_prediction(self, method_name, datasetname,property, dataset_index, model_path):
        modulus_property_map={
            "Young's Modulus":'young', "Shear's Modulus":'shear', "Poisson's Ratio":'poisson'
        }
        dataset_index = int(dataset_index)
        if datasetname == "MetaModulus":
            dataset = LatticeModulus(os.path.join(self.dataset_root,'LatticeModulus'), file_name='data')
        elif datasetname == 'MetaStiffness':
            dataset = LatticeStiffness(os.path.join(self.dataset_root,'LatticeStiffness'), file_name='training')

        if method_name == 'MACE+ve':
            model = MaceVeModel(model_name='mace_ve', dataset_name='LatticeModulus',
                                device=self.device, root_path='./')
            model.config['wandb_args']['use_wandb']=False
            model.config['training']['pred_property'] = modulus_property_map[property]
            model.config['network']['max_edge_radius'] = dataset.edge_feat.max()
            data = dataset.get(dataset_index)
            model.load_model(checkpoint_path=f'./checkpoints/mace_ve/{modulus_property_map[property]}_3/best_model.pth')
            output = model.predict(data)
        return output.view(-1).tolist()

    def method_prediction_result_visualization(self, results, property_name):
        if "Modulus" in property_name or 'modulus' in property_name:
            plot_ellipsoid_colormap_modulus(results, self.default_prediction_path, property_name)

        if 'Poisson' in property_name or 'poisson' in property_name:
            plot_directional_modulus_from_poissons(results,self.default_prediction_path)

        return self.default_prediction_path



    def load_methods_data(self,csv_path):
        self.methods_df = pd.read_csv(csv_path)

        return self.methods_df