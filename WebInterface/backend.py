import os.path
import os
import sys


root_path = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.abspath(os.path.join(root_path, '..'))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)




import pandas as pd
import streamlit as st
from datasets import LatticeModulus
from visualization.homo3D import homo3D
from visualization.vis import visualizeLattice_interactive, visualizeLattice
from visualization.voxel import generate_voxel, visualizeVox
from visualization.visualize_Cij import visualizeCij
import shutil



class ResearchBackend:
    def __init__(self):
        self.datasets = ["LatticeModulus", "LatticeStiffness"]
        self.tasks=['Prediction', 'Generation']
        self.metrics = ["Young_R2", "Young_MAE", "Young_NRMSE"]
        self.dataset_df = None

        self.dataset_root= 'D:\\项目\\Material design\\code_data\\data'

        self.dataset_visualization_root = f'{root_path}/dataset_results'
        self.dataset_visualization_path = os.path.join(self.dataset_visualization_root, 'visualization.png')
        self.dataset_simulation_path = os.path.join(self.dataset_visualization_root, 'simulation.png')
        self.dataset_voxel_path = os.path.join(self.dataset_visualization_root, 'voxel.png')

        self.default_model_interaction_root = f'{root_path}/model_results'
        self.default_model_interaction_path = os.path.join(self.default_model_interaction_root, 'visualization.png')
        self.default_prediction_path = os.path.join(self.default_model_interaction_root, 'prediction.png')

    def load_dataset_stats(self, csv_path):
        """Load dataset statistics from CSV file"""
        self.dataset_df = pd.read_csv(csv_path)
        return self.dataset_df
    
    def get_image_path(self, dataset, metric):
        """Construct image path based on selected options"""
        return f"./data/images/{dataset}_{metric}.png"
    
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
        if datasetname == "MetaModulus":
            dataset = LatticeModulus(os.path.join(self.dataset_root,'LatticeModulus'), file_name='data')
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

    def dataset_info_simulation(self, index, datasetname):
        if datasetname == "MetaModulus":
            dataset = LatticeModulus(os.path.join(self.dataset_root,'LatticeModulus'), file_name='data')
            self.dataset_voxel_path = os.path.join(self.dataset_visualization_root, 'voxel.png')
            self.dataset_simulation_path = os.path.join(self.dataset_visualization_root, 'simulation.png')
            data = dataset[int(index)]
            cart_coords = data.cart_coords.numpy()
            edge_index = data.edge_index.numpy()
            voxel, Density = generate_voxel(10, cart_coords, edge_index, radius=0.1)
            visualizeVox(voxel, self.dataset_voxel_path)
            CH = homo3D(1, 1, 1, 0.5769, 0.3846, voxel)
            plt = visualizeCij(CH, 50)
            plt.savefig(self.dataset_simulation_path)
            plt.close()
        # elif datasetname == 'PointCloud':
        #     self.dataset_visualization_path = os.path.join(self.dataset_visualization_root, 'visualization.png')
        #     shutil.copy(f'{root_path}/data/dataset_info/Cloud_point_samples.png', self.dataset_visualization_path)

    def method_generation(self):
        pass
    def method_prediction(self):
        pass

    def load_methods_data(self,csv_path):
        self.methods_df = pd.read_csv(csv_path)

        return self.methods_df