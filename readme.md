# 1. Project
This project is used for face detection and feature extraction. 
Training is not contained in this project.

# 2. SubProject
## 2.1 Face searching from Video/Image

    python3 -u build_gallery.py --image_path /data/yefei/data/register_photo --save_path /data/yefei/data/register_photo --batch_size 8 --pcnt 5
    nohup python3 -u framework001.py --image_path /data/yefei/data/0107_0114 --save_path /data/yefei/data/0107_0114_toy --gallery_path /data/yefei/data/register_photo/resnet50_irse_mx_model_5932.pth.pkl --toy --pcnt 5 --det_scale 0.5 --rec_weights weights/face_rec/model_5932.pth --det_weights weights/face_det/mobilenet0.25_Final.pth >vid.out 2>&1 &
    nohup python3 -u framework001.py --vid_path /data/yefei/data/video_ucloud/10B_6_1.mp4 --save_path /data/yefei/data/video_ucloud/result --gallery_path /data/yefei/data/register_photo/resnet50_irse_mx_model_5932.pth.pkl --pcnt 5 --det_scale 0.5 >vid.out 2>&1 &

Extract gallery feature from {image_path} and store in {save_path}. 

Searching images or videos via {image_path/vid_path}. 
Result image is composed of original image and two faces from detected face and most similar face from gallery. 
 
# 2.2 Unsupervised face annotation from Video

    nohup python3 -u extract_face_in_video_crontab.py --vid_path /project/data/video_ucloud/ --vid_head 10B --save_path /project/data/video_ucloud_face/ --pcnt 5000 --det_scale 0.5 --det_threshold 0.8 >vid.out 2>&1 &

Extract face images from {vid_path} where each folder is named by its timestamp. 
Each timestamp folder contains serveral folders named by its camera like 10B_6_3. 
10B_6_3 means camera No. 3 from 6th floor inside building No. 10B. 
All face images will be stored in {save_path}. 
A log.txt is used to note which folders are processed, which is used to decide which folders are unprocessed. 

    python -u extract_feature_inside_folder.py --data /project/data/video_ucloud_face --save /project/data/video_ucloud_uncluster_feature --model_name multihead7513

Extract the features of each folder which is generated by method above. 
Each timestamp folder is transferred into a pkl which stores features, image_paths and camera attribution(e.g. 10B_6_3). 
 
    python clustering.py --data /project/data/video_ucloud_uncluster_feature/multihead7513 --save /project/data/video_ucloud_cluster_unsupervised --threshold 1.0 --linkage average --gallery /project/data/register_photo/multihead7513_model_7513.pth.pkl --top 5 --topsim 0.25

Clustering according to the pkls extracted above. 
Each clustered folders contains a bundle of images. 
Basic aggregation(average) is used to generate the feature representing the cluster. 
Then {topsim} is used to decide which cluster will be reserved. 
Only those clusters of which the top1 similarity score is greater than {topsim} is reserved. 
Each cluster folder will also have a folder named top5, where top {top} similar images from gallery will put inside.

# 2.3 SCface database

    python3 SCface_probe_feature_extract.py --data /project/data/SCface/SCface_database/surveillance_cameras_all --save /project/data/SCface/SCface_database/surveillance_cameras_feature --model_name multihead7513 --det_scale 2.0 --det_threshold 0.7
    python3 SCface_gallery_feature_extract.py --data ~/data/SCface/SCface_database/mugshot_frontal_cropped_all/ --save /project/data/SCface/SCface_database/surveillance_cameras_feature --model_name multihead7513 --det_scale 0.5 --det_threshold 0.7

Feature extraction for SCface database. 

