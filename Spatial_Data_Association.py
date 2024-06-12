from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from hungarian import * 
import torch
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from radar_camera_fusion import *
import pprint 


def get_sensor_calibration(calibration_file):
    sensor_calibration_dict = {
        "camera_intrinsics": [],
        "camera_distcoeffs": [],
        "radar_to_camera": [],
        "radar_to_lidar": [],
        "lidar_to_ground": [],
        "camera_to_ground": []
    }

    with open(calibration_file, 'r') as f:
        data = json.load(f)

    for item in data['calibration']:
        if item['calibration'] == 'camera_01':
            sensor_calibration_dict['camera_intrinsics'] = item['k']
            sensor_calibration_dict['camera_distcoeffs'] = item['D']
        elif item['calibration'] == 'radar_01_to_camera_01':
            sensor_calibration_dict['radar_to_camera'] = item['T']
        elif item['calibration'] == 'radar_01_to_lidar_01':
            sensor_calibration_dict['radar_to_lidar'] = item['T']
        elif item['calibration'] == 'lidar_01_to_ground':
            sensor_calibration_dict['lidar_to_ground'] = item['T']
        elif item['calibration'] == 'camera_01_to_ground_homography':
            sensor_calibration_dict['camera_to_ground'] = item['T']
    return sensor_calibration_dict


def homography(list_of_pred_boxes, sensor_calibration_dict):
    ground_coordinate_list = []
    
    for result in list_of_pred_boxes:
        
        bbox = list(result[1:5])

        bottom_center_point = np.array(list(((bbox[2] + bbox[0]) / 2, bbox[3]))).reshape(1, -1) 

        transpose_matrix = np.vstack((np.transpose(bottom_center_point),np.ones((1,1))))
        
        homogeneous_coordinates = np.matmul(sensor_calibration_dict['camera_to_ground'], transpose_matrix)
        ground_coordinates = homogeneous_coordinates / homogeneous_coordinates[-1].reshape(1, -1)

        transpose_ground_coordinates = ground_coordinates.T
        g_x1y1 = transpose_ground_coordinates[0][:2]

        ground_coordinate_list.append([list(g_x1y1)])

    return ground_coordinate_list


def class_box_generator_for_pred(prediction_results):
    for result in prediction_results:
        cls = result.boxes.cls.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        detection = result.boxes.xyxy.cpu().numpy()

        list_of_pred_boxes = np.column_stack((cls, detection, conf))
    
    return list_of_pred_boxes


def radar_to_ground_transfomer(points_array, T_radar_to_lidar, T_lidar_to_ground):

    n_p_array = np.array(points_array).reshape(1,-1)
    tranposed_array = np.transpose(n_p_array)
   
    row_of_ones = np.ones((1, 1))           #1x1
    stacked_matrix = np.vstack((tranposed_array, row_of_ones))  
  
    radar_to_lidar_matrix = np.matmul(T_radar_to_lidar, stacked_matrix)             #3x1

    new_stacked_matrix = np.vstack((radar_to_lidar_matrix, row_of_ones))             #4x1
    in_ground_data = np.matmul(T_lidar_to_ground, new_stacked_matrix)


    in_ground = np.transpose(in_ground_data)

    return in_ground[0]


def radar_to_ground(radar_dict, sensor_calibration_dict):
    
    T = sensor_calibration_dict['radar_to_lidar']
    K = sensor_calibration_dict['lidar_to_ground']

    in_radar = radar_dict
    in_ground = {'cluster': [], 'noise': []}
    for key, value in in_radar.items():
        if key == 'cluster':
            for point in value:
                if point:
                    updated_centroid = radar_to_ground_transfomer(point[0], T, K)
                    updated_lowest_point = radar_to_ground_transfomer(point[1], T, K)
                    updated_velocity = point[2]
                    updated_point = [list(updated_centroid), list(updated_lowest_point), list(updated_velocity)]

                    if key in in_ground:
                        in_ground[key].append(updated_point)
                    else:
                        print('no key exist')
        else:
            for point in value:
                if point:
                    updated_centroid = radar_to_ground_transfomer(point[0], T, K)
                    updated_velocity = [point[1]]
                    updated_point = [list(updated_centroid), list(updated_velocity)]

                    if key in in_ground:
                        in_ground[key].append(updated_point)
                    else:
                        print('no key exist')
                    
    return in_ground


def radar_to_camera_transformer(radar_point, T, k):
   
    n_p_array = np.array(radar_point).reshape(1,-1)
    transpose_RPA = np.transpose(n_p_array)

    new_array = np.vstack([transpose_RPA, np.ones((1, 1))])             
    product_1 = np.matmul(np.array(k), np.array(T))

    product_array = np.matmul(product_1, new_array)                      #[su, sv, s] but along column

    final_array = product_array / product_array [2]                      #[u, v, 1], along column

    u_v = np.delete(final_array, 2, axis = 0)                            #[u, v], along column      
    final_u_v = np.transpose(u_v)

    return final_u_v[0]


def radar_to_camera(radar_output, sensor_calibration_dict):
    T = sensor_calibration_dict['radar_to_camera']
    K = sensor_calibration_dict['camera_intrinsics']
    
    in_radar = radar_output
    in_camera = {'cluster': [], 'noise': []}
    for key, value in in_radar.items():
        if key == 'cluster':
            for point in value:
                if point:
                    updated_centroid = radar_to_camera_transformer(point[0], T, K)
                    updated_lowest_point = radar_to_camera_transformer(point[1], T, K)
                    updated_velocity = point[2]
                    updated_point = [list(updated_centroid), list(updated_lowest_point), list(updated_velocity)]

                    if key in in_camera:
                        in_camera[key].append(updated_point)
                    else:
                        print('no key exist')
        else:
            for point in value:
                if point:
                    updated_centroid = updated_centroid = radar_to_camera_transformer(point[0], T, K)
                    updated_velocity = [point[1]]
                    updated_point = [list(updated_centroid), list(updated_velocity)]

                    if key in in_camera:
                        in_camera[key].append(updated_point)
                    else:
                        print('no key exist')
    
    return in_camera


def get_one_one_association(list_of_pred_boxes, cluster_on_image):
    
    clusters = list(cluster_on_image['cluster']) # Copy of list_of_clusters 
    noise_points = list(cluster_on_image['noise']) 
    pred_boxes = list(list_of_pred_boxes) # Copy of list_of_pred original 
    return_var = 0
    # print(f"Prediction Boxes used: {pred_boxes}")
    # print(f"Clusters used: {clusters}")

    association = {'associated': [], 'non_associated':{'YOLO':[], 'Radar':[]}}

    if len(clusters) > 0 and len(pred_boxes)>0:
        matrix = np.zeros((len(clusters), len(pred_boxes))) 
        # print(matrix)
        for pred_idx, prediction in enumerate(pred_boxes):
            bbox = prediction[1:5]  
            for cluster_idx, cluster in enumerate(clusters):
                cluster_centroid = cluster[0]
                
                if bbox[0] < cluster_centroid[0] < bbox[2] and bbox[1] < cluster_centroid[1] < bbox[3]:
                    matrix[cluster_idx, pred_idx] = 1

                else: 
                    matrix[cluster_idx, pred_idx] = 0
        
        print("Association matrix: :")
        pprint.pprint(matrix)

        special_points = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i,j] == 1:
                    row_sum = sum(matrix[i, k] for k in range(len(matrix[0])))
                    col_sum = sum(matrix[k][j] for k in range(len(matrix)))
                    row_sum = sum(matrix[i,:])
                    col_sum = sum(matrix[:,j]) 

                    if row_sum == 1 and col_sum == 1:
                        special_points.append((i, j)) # Row - Clusters 
                                                      # Column - YOLO

        for item in special_points:
            # print(f"Prediction Index: {item[1]}, Predictions: {pred_boxes}")
            association['associated'].append((pred_boxes[item[1]],clusters[item[0]]))

        for i in range(matrix.shape[0]):
            association["non_associated"]["Radar"].append(clusters[i]) for item in special_points if not in :
            
        for j in range(matrix.shape[1]):
            
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # print(f"i: {i}, j : {j}")
                if (i,j) not in special_points: 
                    association["non_associated"]["Radar"].append(clusters[i])
                    association["non_associated"]["YOLO"].append(pred_boxes[j])

    return association


def main():    

    path_to_images = Path(r'C:\Dk\Projects\Team Project\Dataset\INFRA-3DRC-Dataset\INFRA-3DRC_scene-15\camera_01\camera_01__data')
    path_to_pcd = Path(r'C:\Dk\Projects\Team Project\Dataset\INFRA-3DRC-Dataset\INFRA-3DRC_scene-15\radar_01\radar_01__data')
    image_list = sorted(list(image for image in path_to_images.iterdir()))
    pcd_list = sorted(list(image for image in path_to_pcd.iterdir()))
    yolo_model = YOLO(r"C:\Dk\Projects\Team Project\YOLO detection\Models\Harshit_Large\large_300 epoch_batch 4_augmented\train32\weights\best.pt")
    radar_cluster = r'C:\Dk\Projects\Team Project\Data Association\radar_camera_fusion.py'
    calibration_file = Path(r"C:\Dk\Projects\Team Project\Dataset\INFRA-3DRC-Dataset\INFRA-3DRC_scene-15\calibration.json")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sensor_calibration_dict = get_sensor_calibration(calibration_file)
    # print(sensor_calibration_dict)

    for img, pcd in zip(image_list, pcd_list):

        results = yolo_model.predict(img)
        list_of_pred_boxes = class_box_generator_for_pred(results)
        
        db_scan = my_custom_dbscan(eps1=0.1, eps2=0.5, min_samples=2)
        clusters_on_radar = db_scan.process_pcd_files(pcd)

        image_on_ground = homography(list_of_pred_boxes, sensor_calibration_dict)
        clusters_on_ground = radar_to_ground(clusters_on_radar, sensor_calibration_dict)
        clusters_on_image = radar_to_camera(clusters_on_radar, sensor_calibration_dict)
        # print(f"Prediction boxes: {list_of_pred_boxes}, \n\n\n\ Clusters: {clusters_on_image}")
        data_association = get_one_one_association(list_of_pred_boxes, clusters_on_image) 

        pprint.pprint(f"Dat Assocated : {data_association}")

        # pprint.pprint(f"Data Not Assocated YOLO: {data_association["non_associated"]["YOLO"]}")


if __name__ == '__main__':
    main()