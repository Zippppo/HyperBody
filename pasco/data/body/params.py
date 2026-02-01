"""
Parameters for body semantic segmentation dataset.

71 classes representing different body parts/organs.
"""

import numpy as np

# Number of classes (0-71)
N_CLASSES = 72

# Class names for body parts from Dataset/dataset_info.json
body_class_names = [
    "outside_body",           # 0
    "inside_body_empty",      # 1
    "liver",                  # 2
    "spleen",                 # 3
    "kidney_left",            # 4
    "kidney_right",           # 5
    "stomach",                # 6
    "pancreas",               # 7
    "gallbladder",            # 8
    "urinary_bladder",        # 9
    "prostate",               # 10
    "heart",                  # 11
    "brain",                  # 12
    "thyroid_gland",          # 13
    "spinal_cord",            # 14
    "lung",                   # 15
    "esophagus",              # 16
    "trachea",                # 17
    "small_bowel",            # 18
    "duodenum",               # 19
    "colon",                  # 20
    "adrenal_gland_left",     # 21
    "adrenal_gland_right",    # 22
    "spine",                  # 23
    "rib_left_1",             # 24
    "rib_left_2",             # 25
    "rib_left_3",             # 26
    "rib_left_4",             # 27
    "rib_left_5",             # 28
    "rib_left_6",             # 29
    "rib_left_7",             # 30
    "rib_left_8",             # 31
    "rib_left_9",             # 32
    "rib_left_10",            # 33
    "rib_left_11",            # 34
    "rib_left_12",            # 35
    "rib_right_1",            # 36
    "rib_right_2",            # 37
    "rib_right_3",            # 38
    "rib_right_4",            # 39
    "rib_right_5",            # 40
    "rib_right_6",            # 41
    "rib_right_7",            # 42
    "rib_right_8",            # 43
    "rib_right_9",            # 44
    "rib_right_10",           # 45
    "rib_right_11",           # 46
    "rib_right_12",           # 47
    "skull",                  # 48
    "sternum",                # 49
    "costal_cartilages",      # 50
    "scapula_left",           # 51
    "scapula_right",          # 52
    "clavicula_left",         # 53
    "clavicula_right",        # 54
    "humerus_left",           # 55
    "humerus_right",          # 56
    "hip_left",               # 57
    "hip_right",              # 58
    "femur_left",             # 59
    "femur_right",            # 60
    "gluteus_maximus_left",   # 61
    "gluteus_maximus_right",  # 62
    "gluteus_medius_left",    # 63
    "gluteus_medius_right",   # 64
    "gluteus_minimus_left",   # 65
    "gluteus_minimus_right",  # 66
    "autochthon_left",        # 67
    "autochthon_right",       # 68
    "iliopsoas_left",         # 69
    "iliopsoas_right",        # 70
]
