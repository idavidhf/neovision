# Ocean Structures Laboratory - UFRJ/COPPE
OpenCV-Python scripts for camera calibration in experiments.

Present scripts allow to calibrate stacks of images in loop using chessboard plate

## Under Construction


# Use of the code

#: Script para calibração de imagens
#: Uso: Duas cameras sincronas
#: Revisão: 04/12/2017 (DD/MM/AAAA)

import cv2
import numpy as np
import glob

import matplotlib.pyplot as plt
import tkinter.filedialog

import os

from matplotlib import rc

#: Declare o diretorio onde se encontram os ensaios gravados
swdpath=askdirectory()
os.chdir(swdpath)
#'C:/Users/NEO/Desktop/Ensaio/'

# cwd = os.getcwd() # obter diretorio de trabalho atual


# Coloque os nomes dos ensaios no formato teste= ['teste01','teste02',...,'testeN']
teste=['Amostra_4-1-1','Amostra_4-1-2','Amostra_6-2-1','Amostra_6-3-2']

# As cameras devem ter a mesma clasificação para as imagens sem rectificar como para as imagens de calibração
imgs = ['Frontal','Lateral']

for val in range(0,len(teste),1):
    #cwd = os.getcwd() #::: Verificar se o diretorio já fica como local
    bpth = teste[val]+'/'

    for i in range(1,len(imgs),1):
        # Pasta com as imagens de calibração (considerase 'Calibracao' como a pasta principal)
        pth = bpth+'Calibracao/'+imgs[i]+'/'

        # Folder with videos to be calibrated
        pthtorectify = bpth+imgs[i]+'/'

'''
        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 12}
        rc('font', **font)
        '''
        # Critério de terminação do mapeamento dos pontos no tablero de calibração
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Preparação dos campos para os pontos, ex. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32) # Tablero xadrez 6x9 (H,V)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        # Gera um array com os nomes de todas as imagens na pasta com extensão '*.jpg','*.png',etc.
        images = glob.glob(pth+'*.jpg')
        
        #Procure e guarde os opntos encontrados nas imagens de calibração
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
            print(fname)
                # Draw and display the corners
                #img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
                #cv2.imshow('img',img)
                #cv2.waitKey(500)

        cv2.destroyAllWindows() #::: Testar omitindo esta linha
        
        # Calibra a câmera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        
        # Rectifica as imagens usando a calibração da cÂmera anterior
        # ->->-> Calibration by cv2.undistort() OpenCV-Py library
        #: 1 - Load and image
        img = cv2.imread(images[1])
        #: 2 - Get the shape of loaded image
        h,  w = img.shape[:2]
        #: 3 - Estimate the camera parameters
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        
        imgtorectify = glob.glob(pthtorectify+'*.jpg')
        
        pthout = pthtorectify + 'Undistorted_images/'

        if not os.path.exists(pthout):
            os.mkdir(pthout)
            
        
        for val in imgtorectify:
            img0 = cv2.imread(val)
            # undistort
            dst = cv2.undistort(img0, mtx, dist, None, newcameramtx)

            # crop the image
            #x, y, w, h = roi
            #dst = dst[y:y + h, x:x + w]
            cv2.imwrite(pthout+val[len(pthtorectify):-4]+'.jpg', dst)
            print(pthout + val[len(pthtorectify):-4] + '.jpg')

print("Calibração completada com sucesso")

