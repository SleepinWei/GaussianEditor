# DATASET_ROOT=/DATA1/zhuyunwei/OpenDataLab___Mill_19/raw/Mill_19/rubble-pixsfm/chunks

DATASET_PATH=/DATA1/zhuyunwei/OpenDataLab___Mill_19/raw/Mill_19/rubble-pixsfm

# set -e 
# for idx in $(seq 0 8);
# do 
#     DATASET_PATH=$DATASET_ROOT/$idx
#     colmap feature_extractor --database_path $DATASET_PATH/database.db \
#         --image_path $DATASET_PATH/images

#     colmap exhaustive_matcher \
#        --database_path $DATASET_PATH/database.db

#     mkdir $DATASET_PATH/sparse

#     colmap mapper \
#         --database_path $DATASET_PATH/database.db \
#         --image_path $DATASET_PATH/images\
#         --output_path $DATASET_PATH/sparse
# done
# convert bin to txt
# colmap model_converter --input_path /DATA1/zhuyunwei/OpenDataLab___Mill_19/raw/Mill_19/rubble-pixsfm/train/sparse/0 --output_path /DATA1/zhuyunwei/OpenDataLab___Mill_19/raw/Mill_19/rubble-pixsfm/train/sparse/0 --output_type TXT
# colmap automatic_reconstructor \
    # --workspace_path $DATASET_PATH \
    # --image_path $DATASET_PATH/rgbs \
    # --gpu_index 1

# export CUDA_VISIBLE_DEVICES=3;

# colmap feature_extractor --database_path $DATASET_PATH/database.db --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.camera_params "2977.529,2977.529,2304.0,0" \
    # --image_path $DATASET_PATH/train/rgbs --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1  

# colmap feature_extractor --database_path $DATASET_PATH/database.db --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index 3\
#     --image_path $DATASET_PATH/train/rgbs # --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1  

# colmap exhaustive_matcher\
#        --database_path $DATASET_PATH/database.db \
#        --SiftMatching.guided_matching 1 \
#        --TwoViewGeometry.confidence 0.9

# mkdir $DATASET_PATH/sparse
# mkdir $DATASET_PATH/sparse/0

# rm $DATASET_PATH/train/sparse/0/*

colmap point_triangulator \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/train/rgbs \
    --input_path $DATASET_PATH/train/sparse \
    --output_path $DATASET_PATH/train/sparse/0 \
    --Mapper.tri_ignore_two_view_tracks 0 \
    # --Mapper.tri_min_angle 0.000000000001 \
    # --Mapper.init_num_trials 500 \
    # --Mapper.ba_refine_focal_length 0  \
    # --Mapper.ba_refine_extra_params 0 \
    # --Mapper.local_ba_min_tri_angle 0.000001 \
    # --Mapper.tri_re_max_trials 10 \
    # --Mapper.ba_local_num_images 100 \
    # --Mapper.filter_min_tri_angle 0.00000001 


# colmap image_undistorter \
#     --image_path $DATASET_PATH/train/rgbs\
#     --input_path $DATASET_PATH/train/sparse\
#     --output_path $DATASET_PATH/train/dense

# colmap patch_match_stereo \
#     --workspace_path $DATASET_PATH/train/dense

# colmap stereo_fusion \
#     --workspace_path $DATASET_PATH/train/dense\
#     --output_path $DATASET_PATH/train/dense/fused.ply