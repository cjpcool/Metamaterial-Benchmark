import base64

import streamlit as st
from backend import ResearchBackend
import os
import pandas as pd


# current directory
root_path = os.path.dirname(os.path.abspath(__file__))
# Initialize backend
backend = ResearchBackend()

# Configure page
st.set_page_config(page_title="MetamatBench", layout="wide")

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'Page 1'
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = None

# Common title
st.title("MetamatBench")

# Navigation
page_options = {
    "Page 1": "Model Selection",
    "Page 2": "Dataset Analytics",
    "Page 3": "Human-AI Collaboration",
}
selected_page = st.sidebar.selectbox("Navigation", list(page_options.keys()), format_func=lambda x: page_options[x])


def read_methods_csv(filepath):
    try:
        df = backend.load_methods_data(filepath)
        # Convert the URL column to clickable links with a PDF icon
        if 'URL' in df.columns:
            df['URL'] = df['URL'].apply(lambda url: f'<a href="{url}" target="_blank">📄</a>')
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None


# Page 1: Ranking Board
def show_page_1():
    st.header("Model Selection")
    st.subheader("Ranking Board")
    # Create columns for dropdowns
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset = st.selectbox("Datasets", backend.datasets)
    with col2:
        task = st.selectbox("Tasks", backend.tasks)
        if task == 'Prediction':
            metrics = backend.prediction_results.columns[2:-1]
        else:
            metrics = backend.generation_results.columns[2:-1]
    with col3:
        metric = st.selectbox("Metrics", metrics)

    # Display image
    image_path = backend.get_image_path(dataset, task, metric)
    try:
        st.image(image_path, use_container_width=True)
    except FileNotFoundError:
        st.error("Image not found for selected combination")

    # Display generated content
    # description = backend.get_metric_description(dataset, metric)
    st.markdown("### Model Details")
    # st.markdown(description)

    # Read the CSV file and display the table with clickable URL links
    methods_df = read_methods_csv(f"{root_path}/data/methods.csv")
    if methods_df is not None:
        # Convert DataFrame to HTML (disable escaping so our HTML links render)
        html_table = methods_df.to_html(escape=False, index=False)
        st.markdown(html_table, unsafe_allow_html=True)


# Page 2: Datasets
def show_page_2():
    st.header(page_options['Page 2'])
    if "dataset_results_cleared" not in st.session_state:
        # backend.clear_dataset_results()
        st.session_state["dataset_results_cleared"] = True

    # st.header("Dataset Analytics")

    # M1: Visualization / Voxel / Simulation Module
    st.subheader("Visualization / Voxel / Simulation")
    cols_m1 = st.columns(3)
    viz_placeholder = cols_m1[0].empty()
    voxel_placeholder = cols_m1[1].empty()
    sim_placeholder = cols_m1[2].empty()

    def display_viz(viz_path):
        if os.path.exists(viz_path):
            with open(viz_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            html = f"""
            <div style="height:30vh; display: flex; align-items: center; justify-content: center;">
                <img alt="Visualization Results" src="data:image/png;base64,{encoded_image}" style="max-height:100%; width:auto;" />
            </div>
            """
            viz_placeholder.markdown(html, unsafe_allow_html=True)
        else:
            viz_placeholder.info("No visualization result available yet. Simulation may take several minutes.")

    def display_voxel(voxel_path):
        if os.path.exists(voxel_path):
            with open(voxel_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            html = f"""
            <div style="height:30vh; display: flex; align-items: center; justify-content: center;">
                <img alt="Voxel Results" src="data:image/png;base64,{encoded_image}" style="max-height:100%; width:auto;" />
            </div>
            """
            voxel_placeholder.markdown(html, unsafe_allow_html=True)
        else:
            voxel_placeholder.info("No voxel result available yet. Simulation may take several minutes.")

    def display_sim(sim_path):
        if os.path.exists(sim_path):
            with open(sim_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            html = f"""
            <div style="height:30vh; display: flex; align-items: center; justify-content: center;">
                <img alt="Simulation Results" src="data:image/png;base64,{encoded_image}" style="max-height:100%; width:auto;" />
            </div>
            """
            sim_placeholder.markdown(html, unsafe_allow_html=True)
        else:
            sim_placeholder.info("No simulation result available yet. Simulation may take several minutes.")

    display_viz(backend.dataset_visualization_path)
    display_voxel(backend.dataset_voxel_path)
    display_sim(backend.dataset_simulation_path)

    # M2: Data Selection Module
    if st.session_state.get("selected_dataset"):
        st.subheader("Data Selection")
        st.markdown(f"Detailed information for dataset: **{st.session_state.selected_dataset}**. (Only MetaModulus and MetaStiffness supproted currently.)")
        index_input = st.text_input("Enter data index", key="data_index",placeholder="Enter sample index [0,1,2,...,N-1]")
        col_viz, = st.columns(1)
        if col_viz.button("Visualize Unit Cell"):
            new_viz_path = backend.dataset_info_visualize(index_input, st.session_state.selected_dataset)
            display_viz(new_viz_path)
        voxel_size_input = st.text_input("Enter voxel size", key="vox_size", placeholder="Enter voxel size (int, 30-40)")

        col_sim, = st.columns(1)
        if col_sim.button("Simulate"):
            vox, new_voxel_path = backend.dataset_info_vox(index_input, voxel_size_input, st.session_state.selected_dataset)
            display_voxel(new_voxel_path)
            new_sim_path = backend.dataset_info_simulation(vox)
            display_sim(new_sim_path)

            # new_voxel_path, new_sim_path = backend.dataset_info_vox_and_simulation(index_input, voxel_size_input, st.session_state.selected_dataset)
            # display_sim(new_sim_path)

    # M3: Dataset Statistics Module
    st.subheader("Dataset Statistics")
    dataset_df = backend.load_dataset_stats(f"{root_path}/data/dataset_stats.csv")
    num_cols = len(dataset_df.columns)
    header_cols = st.columns(num_cols)
    for i, col_name in enumerate(dataset_df.columns):
        header_cols[i].markdown(f"**{col_name}**")
    for i, row in dataset_df.iterrows():
        cols = st.columns(num_cols)
        dataset_name = row[dataset_df.columns[0]]
        button_key = f"ds_{dataset_name}"
        if cols[0].button(dataset_name, key=button_key):
            if i < 3:
                st.session_state.selected_dataset = dataset_name
        for col, val in zip(cols[1:], row[1:]):
            col.write(val)


def show_page_3():
    st.header(page_options['Page 3'])

    is_prediction = False
    if st.session_state.get("selected_method"):
        method_info = st.session_state.selected_method_info
        if method_info["Task"] == "Prediction":
            is_prediction = True

    # M1: Results Module
    if is_prediction:
        st.subheader("Results Visualization")
        cols_result = st.columns(2)
        interaction_placeholder = cols_result[0].empty()
        prediction_placeholder = cols_result[1].empty()

        def display_interaction(result_path):
            if os.path.exists(result_path):
                with open(result_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode()
                html = f"""
                <div style="height:30vh; display: flex; align-items: center; justify-content: center;">
                    <img alt="Interaction Result" src="data:image/png;base64,{encoded_image}" style="max-height:100%; width:auto;" />
                </div>
                """
                interaction_placeholder.markdown(html, unsafe_allow_html=True)
            else:
                interaction_placeholder.info("No interaction result available yet.")

        def display_prediction(result_path):
            if os.path.exists(result_path):
                with open(result_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode()
                html = f"""
                <div style="height:30vh; display: flex; align-items: center; justify-content: center;">
                    <img alt="Prediction Result" src="data:image/png;base64,{encoded_image}" style="max-height:100%; width:auto;" />
                </div>
                """
                prediction_placeholder.markdown(html, unsafe_allow_html=True)
            else:
                prediction_placeholder.info("No prediction result available yet.")

        # display_interaction(backend.default_model_interaction_path)
        # display_prediction(backend.default_prediction_path)
    else:
        st.subheader("Results Visualization")
        result_placeholder = st.empty()

        def display_result(result_path):
            if os.path.exists(result_path):
                with open(result_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode()
                html = f"""
                <div style="height:30vh; display: flex; align-items: center; justify-content: center;">
                    <img alt="Model Interaction Result" src="data:image/png;base64,{encoded_image}" style="max-height:100%; width:auto;" />
                </div>
                """
                result_placeholder.markdown(html, unsafe_allow_html=True)
            else:
                result_placeholder.info("No result available yet.")

        display_result(backend.default_model_interaction_path)

    # M2: Data Selection Module
    if st.session_state.get("selected_method"):
        method_info = st.session_state.selected_method_info
        st.subheader("Interaction Configuration")
        st.markdown(f"Detailed information for method: **{method_info['Method']}**. (Only MetaModulus supported currently.)")
        task = method_info["Task"]
        if task == "Generation":
            # Generation
            dataset = st.selectbox("Dataset", backend.datasets, key="gen_dataset")
            model_path = st.text_input("Model Path", key="pred_model_path", disable=True, value='checkpoint')
            save_path = st.text_input("Save Path", key="gen_save_path")
            condition_value = st.text_input("Condition Value", key="gen_condition_value", placeholder="Can be null")
            col_gen, _ = st.columns(2)
            if col_gen.button("Generate"):
                result_path = backend.method_generation(dataset, model_path, save_path, condition_value)
                if is_prediction:
                    display_interaction(result_path)
                else:
                    display_result(result_path)
        elif task == "Prediction":
            # dataset = st.selectbox("Dataset", backend.datasets, key="pred_dataset")
            properties = ["Young's Modulus", "Shear's Modulus", "Poisson's Ratio"]
            col1, col2 = st.columns(2)
            with col1:
                dataset = st.selectbox("Datasets", backend.datasets)
            with col2:
                property = st.selectbox("Property", properties)
            dataset_index = st.text_input("Dataset Index", key="pred_dataset_index")
            col_viz, = st.columns(1)
            if col_viz.button("Visualize Unit Cell"):
                st.session_state.interaction_data_vis_new_path = backend.interaction_data_visualize(dataset_index, dataset)
                display_interaction(st.session_state.interaction_data_vis_new_path)

            model_path = st.text_input("Model Path", key="pred_model_path", disabled=True, value='checkpoint')
            col_pred, _ = st.columns(2)
            if col_pred.button("Predict"):
                if 'interaction_data_vis_new_path' not in st.session_state.keys():
                    st.session_state.interaction_data_vis_new_path = backend.interaction_data_visualize(dataset_index,
                                                                                                        dataset)
                display_interaction(st.session_state.interaction_data_vis_new_path)
                prediction_result = backend.method_prediction(st.session_state.selected_method, dataset, property, dataset_index, model_path)
                st.markdown(f"Predict {property}: {prediction_result}.")
                prediction_result_vis_path = backend.method_prediction_result_visualization(prediction_result, property)
                display_prediction(prediction_result_vis_path)
        else:
            st.info("Unsupported task type for this method.")

    # M3: Methods Module
    st.subheader("Methods")
    # methods_df = backend.load_methods_data(f"{root_path}/data/demo_methods.csv")
    methods_df = read_methods_csv(f"{root_path}/data/demo_methods.csv")
    num_cols = len(methods_df.columns)
    header_cols = st.columns(num_cols)
    for i, col_name in enumerate(methods_df.columns):
        header_cols[i].markdown(f"**{col_name}**")
    for _, row in methods_df.iterrows():
        cols = st.columns(num_cols)
        method_name = row["Method"]
        button_key = f"method_{method_name}"
        if cols[0].button(method_name, key=button_key):
            st.session_state.selected_method = method_name
            st.session_state.selected_method_info = row.to_dict()  # 将当前行数据存入 session_state 供 M2 使用

        for col, val in zip(cols[1:], row[1:]):
            if  isinstance(val, str) and ('https' in val or 'http' in val):
                col.markdown( val,  unsafe_allow_html=True)
            # st.markdown(html_table, unsafe_allow_html=True)
            else:
                col.write(val)


# Show selected page
if selected_page == "Page 1":
    show_page_1()
elif selected_page =='Page 2':
    show_page_2()
elif selected_page =='Page 3':
    show_page_3()