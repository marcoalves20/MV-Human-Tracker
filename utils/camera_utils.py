import os
import pickle
import numpy as np

def load_calibration(path):
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    cameras = {}
    for file in files:
        camera = file[:-4]
        cam = pickle.load(open(path+file, 'rb'))
        cameras[camera] = {'K': cam['K'], 'R': cam['R'], 't': cam['t'], 'dist': np.array([1,1,1,0,0])}
    return cameras


if __name__ == '__main__':
    test = load_calibration('../calibration/')
    print(test)
