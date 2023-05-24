# Create a conda virtual environment and activate it
conda create -n aff python=3.8
conda activate aff

# Install requirements
pip install \
        yacs==0.1.8 \
        termcolor==2.2.0 \
        timm==0.6.12 \
        pykeops==2.1.1 \
        ptflops==0.6.9 \
        numpy==1.22.4 \
        cython==0.29.33 \
        scipy==1.9.1 \
        shapely==2.0.1 \
        h5py==3.8.0 \
        submitit==1.4.5 \
        scikit-image==0.20.0
conda install -c conda-forge opencv
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

# Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# add ADE20K_SEM_SEG_CATEGORIES_COLORS for consistent color in ADE prediction visualization
mv ./builtin.py path/to/conda/lib/python3.8/site-packages/detectron2/data/datasets
mv ./builtin_meta.py path/to/conda/lib/python3.8/site-packages/detectron2/data/datasets

# Install the custom CUDA kernels for AFF
cd mask2former/modeling/clusten/src && python setup.py install
