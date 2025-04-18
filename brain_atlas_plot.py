############################################################
# Visualizing Brain Atlas Data with Statistical Mapping and Custom Plotting
# This script loads data from a CSV file, processes it, and generates various visualizations
# of the Destrieux brain atlas using nilearn, with different plotting methods like 
# statistical maps, interactive views, glass brain, surface projections, and 3D browser.
############################################################
import pandas as pd
import numpy as np
import nilearn.plotting as plotting
from nilearn.datasets import fetch_atlas_destrieux_2009
from nilearn import image
import matplotlib.pyplot as plt
# https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_atlas_destrieux_2009.html#nilearn.datasets.fetch_atlas_destrieux_2009
# https://nilearn.github.io/dev/auto_examples/01_plotting/plot_3d_map_to_surface_projection.html#sphx-glr-auto-examples-01-plotting-plot-3d-map-to-surface-projection-py

##################################################
# Step 1: Define Plotting Parameters and Fetch Atlas
##################################################

plotting_method = 'glass_brain' # 'statistical_map', 'interactive', 'glass_brain', 'surface_projection', '3d_browser'
cmap = 'coolwarm'  # Change cmap if needed, try 'hot', 'coolwarm', or 'inferno'

# Fetch the Destrieux atlas
atlas = fetch_atlas_destrieux_2009(lateralized=True)

# Get the atlas maps and labels
atlas_maps = atlas['maps']
atlas_labels = atlas['labels']

# Set the directory path for the output
output_dir = '/Users/fvashegh/Library/CloudStorage/OneDrive-JNJ/GAMAL/results/'

#%%
##################################################
# Step 2.1: Load and Clean Data
# - Load CSV, filter rows, and clean necessary columns.
##################################################

# Load the CSV file into a pandas DataFrame
file_path = '/Users/fvashegh/Library/CloudStorage/OneDrive-JNJ/GAMAL/Sig_results_T1_Combined_5Regions.csv'
data_df = pd.read_csv(file_path)

# Define the subcortical region and subtype (all user-defined variables)
subcortical_region = 'accumbens'  # subcortical region name --> amygdala, hippocampus, accumbens
subtype = 'C3'  # Change this to any other subtype value, such as 'C1', 'C3', etc.

# Filter and clean the data in one step
data_df_clean = (
    data_df[
        # Filter based on subcortical region and subtype
        (data_df['idp_aseg'].str.contains(subcortical_region, case=False, na=False)) &
        (data_df['subtype'] == subtype)
    ]
    [['estimate', 'idp_fs2009']]  # Select only the necessary columns
    .assign(
        # Clean the 'idp_fs2009' column by removing prefixes and modifying hemisphere names
        idp_fs2009=lambda x: x['idp_fs2009'].str.replace(
            r'^(Freesurfer_a2009s_(area_of_|mean_thickness_of_|volume_of_))', '', regex=True
        ).apply(
            # Modify hemisphere naming: prepend 'R ' or 'L ' based on the hemisphere
            lambda y: 'R ' + y.replace('_right_hemisphere', '') if '_right_hemisphere' in y 
            else ('L ' + y.replace('_left_hemisphere', '') if '_left_hemisphere' in y else y)
        )
    )
    .dropna()  # Remove rows with missing values
)

# Check the cleaned data
print(data_df_clean.head())

#%%
##################################################
# Step 2.2: Load and Clean Data
# - Load CSV, filter rows, and clean necessary columns.
##################################################

# Load the CSV file into a pandas DataFrame
file_path = '/Users/fvashegh/Library/CloudStorage/OneDrive-JNJ/GAMAL/SCA4_COMB_RESULTS_T1.csv'
data_df = pd.read_csv(file_path)

# Define the subcortical region and subtype (all user-defined variables)
subcortical_region = 'amygdala'  # Change this to any other subcortical region name --> amygdala, accumbens
subtype = 'C2'  # Change this to any other subtype value, such as 'C1', 'C3', etc.

# Define the measurement type to keep: 'mean', 'area', or 'volume'
measurement_type = 'mean'  # Change this to 'area' or 'volume' as needed

# Filter and clean the data based on the selected measurement type
data_df_clean = (
    data_df
    [['fs', 'Est_2L']]  # Select only the necessary columns
    .loc[lambda x: x['fs'].str.contains(measurement_type)]  # Filter rows based on measurement type
    .assign(
        # Clean the 'fs' column by removing prefixes and modifying hemisphere names
        fs=lambda x: x['fs'].str.replace(
            r'^(Freesurfer_a2009s_(area_of_|mean_thickness_of_|volume_of_))', '', regex=True
        ).apply(
            # Modify hemisphere naming: prepend 'R ' or 'L ' based on the hemisphere
            lambda y: 'R ' + y.replace('_right_hemisphere', '') if '_right_hemisphere' in y 
            else ('L ' + y.replace('_left_hemisphere', '') if '_left_hemisphere' in y else y)
        )
    )
    .dropna()  # Remove rows with missing values
    .rename(columns={'fs': 'idp_fs2009', 'Est_2L': 'estimate'})  # Rename columns
)

# Check the cleaned data
print(data_df_clean.head())

#%%
##################################################
# Step 3: Process Data for Visualization
##################################################

# Initialize plot_data as zeros
num_regions = len(atlas_labels) - 1  # Exclude background
plot_data = np.zeros(num_regions)
# plot_data = 2 * np.random.rand(num_regions) - 1  # random values between -1 and 1

# Loop through each row in measure_df_clean
for _, row in data_df_clean.iterrows():
    search_string = row['idp_fs2009'].replace('-', '_').lower()
    estimate_value = row['estimate']
    
    # Find the matching region and set the corresponding plot_data value (case-insensitive and hyphen/underscore flexible)
    match_index = atlas_labels[atlas_labels["name"].str.replace('-', '_').str.lower() == search_string].index
    if not match_index.empty:
        plot_data[match_index[0] - 1] = estimate_value  # -1 to account for 0-indexing

##################################################
# Step 4: Create and Save Visualization
##################################################

# Create a new image where each region is assigned its random value
new_img = image.math_img("img", img=atlas_maps)
for i in range(1, num_regions + 1):  # Labels start from 1
    new_img = image.math_img("(img == {0}) * {1} + (img != {0}) * img".format(i, plot_data[i - 1]), img=new_img)

# Set the path for the HTML or PDF output
if plotting_method in ["3d_browser", "interactive"]:
    save_path = f'{output_dir}destrieux_{subcortical_region}_{plotting_method}.html'
else:
    save_path = f'{output_dir}destrieux_{subcortical_region}_{plotting_method}.pdf'

# Plot based on the defined method
if plotting_method == "statistical_map":
    plotting.plot_stat_map(
        stat_map_img=new_img,
        display_mode="ortho", 
        # display_mode options: "ortho" (3 views), "x" (sagittal), "y" (coronal), "z" (axial),  
        # "yx" (coronal+sagittal), "xz" (axial+sagittal), "yz" (axial+coronal), "mosaic" (grid layout).
        #cut_coords=range(0, 51, 10),  # Show slices at intervals of 10
        #cut_coords=(0, 0, 0),
        threshold=0.5,
        #title="Destrieux Atlas",
        cmap=cmap,
        colorbar=True
    )
    # Save the plot to a PDF file using matplotlib's savefig
    plt.savefig(save_path)
    plotting.show()

elif plotting_method == "interactive":
    # Use view_img for an interactive 3D visualization
    view = plotting.view_img(new_img, title='Destrieux Atlas', cmap=cmap)
    # Save/view the interactive HTML file
    view.save_as_html(save_path)  # Interactive plots are saved as HTML, not PDF
    view.open_in_browser() 
    
elif plotting_method == "glass_brain":
    # Plot the modified atlas with random values in a glass brain view
    plotting.plot_glass_brain(
        stat_map_img=new_img,
        display_mode="ortho",  # "r" shows right hemisphere; options: "ortho", "x", "y", "z", "l", "r"
        plot_abs=False,  # Do not take the absolute value of the data
        #title="Destrieux Atlas",
        threshold=0,
        cmap=cmap,
        colorbar=True,
        vmin=-25, # 15, 17, 25
        vmax=+25
    )
    # Save the plot to a PDF file using matplotlib's savefig
    plt.savefig(save_path)
    plotting.show()
    

elif plotting_method == "surface_projection":
    # Plot the image on the surface of the brain with multiple views and hemispheres
    plotting.plot_img_on_surf(
        stat_map=new_img,
        #surf_mesh="fsaverage", # fsaverage: high-res
        #views=['anterior', 'posterior', 'medial', 'lateral', 'dorsal', 'ventral'],
        views=['lateral', 'medial'],  # Lateral and medial views
        # Supported values are: ('anterior', 'posterior', 'medial', 'lateral', 'dorsal', 'ventral') 
        # or a sequence of length 2 setting the elevation and azimut of the camera.
        hemispheres=["left", "right"],  # Plot for both hemispheres
        #title="Destrieux Atlas",
        cmap=cmap,
        threshold=0.5,  # Optional threshold (or None)
        bg_on_data=True  # Background on data
    )
    # Save the plot to a PDF file using matplotlib's savefig
    plt.savefig(save_path)
    plotting.show()

elif plotting_method == "3d_browser":
    # Create 3D visualization using view_img_on_surf with improved options
    view = plotting.view_img_on_surf(
        stat_map_img=new_img,
        surf_mesh="fsaverage", # fsaverage, fsaverage3-7
        cmap=cmap,
        threshold="0%",
        vol_to_surf_kwargs={
            "n_samples": 10,  # More sampling points for better projection
            "radius": 0.0,  # Adjust if needed for better coverage
            "interpolation": "linear",
        },
        symmetric_cmap=True,  # Keep colormap symmetric for positive & negative values
        colorbar=False,
    )
    # Save/view the interactive HTML file
    view.save_as_html(save_path)
    view.open_in_browser() 

else:
    print("Invalid plotting method specified.")



















#%%
############################################################
# Visualizing Brain Atlas Data with Statistical Mapping and Custom Plotting
# This script loads data from a CSV file, processes it, and generates various visualizations
# of the Destrieux brain atlas using nilearn, with different plotting methods like 
# statistical maps, interactive views, glass brain, surface projections, and 3D browser.
############################################################
import numpy as np
import nilearn.plotting as plotting
from nilearn.datasets import fetch_atlas_destrieux_2009
from nilearn import image
import matplotlib.pyplot as plt
# https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_atlas_destrieux_2009.html#nilearn.datasets.fetch_atlas_destrieux_2009
# https://nilearn.github.io/dev/auto_examples/01_plotting/plot_3d_map_to_surface_projection.html#sphx-glr-auto-examples-01-plotting-plot-3d-map-to-surface-projection-py

##################################################
# Step 1: Define Plotting Parameters and Fetch Atlas
##################################################

plotting_method = '3d_browser' # 'statistical_map', 'interactive', 'glass_brain', 'surface_projection', '3d_browser'
cmap = 'coolwarm'  # Change cmap if needed, try 'hot', 'coolwarm', or 'inferno'

# Fetch the Destrieux atlas
atlas = fetch_atlas_destrieux_2009(lateralized=True)

# Get the atlas maps and labels
atlas_maps = atlas['maps']
atlas_labels = atlas['labels']

# Set the directory path for the output
output_dir = 'path/to/your/directory/'

##################################################
# Step 2: Load and Clean Data
# - Load CSV, filter rows, and clean necessary columns.
##################################################

# Load the data
file_path = "path_to_your_file.csv"
data_df = pd.read_csv(file_path)

# Define filter conditions (customizable)
filter_condition = 'example_condition'

# Clean and filter data
cleaned_data = (
    data_df[data_df['column_name'].str.contains(filter_condition, na=False)]
    [['column1', 'column2']]
    .dropna()  # Remove missing values
)

# View cleaned data
print(cleaned_data.head())

##################################################
# Step 3: Process Data for Visualization
##################################################

# Initialize plot_data as zeros
num_regions = len(atlas_labels) - 1  # Exclude background
plot_data = np.zeros(num_regions)
# plot_data = 2 * np.random.rand(num_regions) - 1  # random values between -1 and 1

# Loop through each row in measure_df_clean
for _, row in data_df_clean.iterrows():
    search_string = row['idp_fs2009'].replace('-', '_').lower()
    estimate_value = row['estimate']
    
    # Find the matching region and set the corresponding plot_data value (case-insensitive and hyphen/underscore flexible)
    match_index = atlas_labels[atlas_labels["name"].str.replace('-', '_').str.lower() == search_string].index
    if not match_index.empty:
        plot_data[match_index[0] - 1] = estimate_value  # -1 to account for 0-indexing

##################################################
# Step 4: Create and Save Visualization
##################################################

# Create a new image where each region is assigned its random value
new_img = image.math_img("img", img=atlas_maps)
for i in range(1, num_regions + 1):  # Labels start from 1
    new_img = image.math_img("(img == {0}) * {1} + (img != {0}) * img".format(i, plot_data[i - 1]), img=new_img)

# Set the path for the HTML or PDF output
if plotting_method in ["3d_browser", "interactive"]:
    save_path = f'{output_dir}destrieux_{subcortical_region}_{plotting_method}.html'
else:
    save_path = f'{output_dir}destrieux_{subcortical_region}_{plotting_method}.pdf'

# Plot based on the defined method
if plotting_method == "statistical_map":
    plotting.plot_stat_map(
        stat_map_img=new_img,
        display_mode="mosaic", 
        # display_mode options: "ortho" (3 views), "x" (sagittal), "y" (coronal), "z" (axial),  
        # "yx" (coronal+sagittal), "xz" (axial+sagittal), "yz" (axial+coronal), "mosaic" (grid layout).
        #cut_coords=range(0, 51, 10),  # Show slices at intervals of 10
        #cut_coords=(0, 0, 0),
        threshold=0.5,
        #title="Destrieux Atlas",
        cmap=cmap,
        colorbar=True
    )
    # Save the plot to a PDF file using matplotlib's savefig
    plt.savefig(save_path)
    plotting.show()

elif plotting_method == "interactive":
    # Use view_img for an interactive 3D visualization
    view = plotting.view_img(new_img, title='Destrieux Atlas', cmap=cmap)
    # Save/view the interactive HTML file
    view.save_as_html(save_path)  # Interactive plots are saved as HTML, not PDF
    view.open_in_browser() 
    
elif plotting_method == "glass_brain":
    # Plot the modified atlas with random values in a glass brain view
    plotting.plot_glass_brain(
        stat_map_img=new_img,
        display_mode="ortho",  # "r" shows right hemisphere; options: "ortho", "x", "y", "z", "l", "r"
        plot_abs=False,  # Do not take the absolute value of the data
        #title="Destrieux Atlas",
        threshold=0.5,
        cmap=cmap,
        colorbar=True
    )
    # Save the plot to a PDF file using matplotlib's savefig
    plt.savefig(save_path)
    plotting.show()
    

elif plotting_method == "surface_projection":
    # Plot the image on the surface of the brain with multiple views and hemispheres
    plotting.plot_img_on_surf(
        stat_map=new_img,
        views=["lateral", "medial"],  # Lateral and medial views
        hemispheres=["left", "right"],  # Plot for both hemispheres
        #title="Destrieux Atlas",
        cmap=cmap,
        threshold=0.5,  # Optional threshold (or None)
        bg_on_data=True  # Background on data
    )
    # Save the plot to a PDF file using matplotlib's savefig
    plt.savefig(save_path)
    plotting.show()

elif plotting_method == "3d_browser":
    # Create 3D visualization using view_img_on_surf with improved options
    view = plotting.view_img_on_surf(
        stat_map_img=new_img,
        surf_mesh="fsaverage", # fsaverage, fsaverage3-7
        cmap=cmap,
        threshold="0%",
        vol_to_surf_kwargs={
            "n_samples": 10,  # More sampling points for better projection
            "radius": 0.0,  # Adjust if needed for better coverage
            "interpolation": "linear",
        },
        symmetric_cmap=True,  # Keep colormap symmetric for positive & negative values
        colorbar=False,
    )
    # Save/view the interactive HTML file
    view.save_as_html(save_path)
    view.open_in_browser() 

else:
    print("Invalid plotting method specified.")


















