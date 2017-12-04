# Ocean Structures Laboratory - UFRJ/COPPE
OpenCV-Python scripts for camera calibration in experiments.
Present scripts allow to calibrate stacks of images in loop using chessboard plate

----------
#### Use of the code

Script para calibração de imagens
Uso: Duas cameras sincronas
Revisão: 04/12/2017 (DD/MM/AAAA)
----------

    import cv2
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import tkinter.filedialog
	import os
	from matplotlib import rc
   
Declare o diretorio onde se encontram os ensaios gravados

	swdpath=askdirectory()
	os.chdir(swdpath)

Coloque os nomes dos ensaios num array

	teste=['Amostra_4-1-1','Amostra_4-1-2','Amostra_6-2-1','Amostra_6-3-2']

As cameras devem ter a mesma clasificação para as imagens sem rectificar como para as imagens de calibração

	imgs = ['Frontal','Lateral']
	for val in range(0,len(teste),1):
	       bpth = teste[val]+'/'

	    for i in range(1,len(imgs),1):
            pth = bpth+'Calibracao/'+imgs[i]+'/'
            # Folder with videos to be calibrated
            pthtorectify = bpth+imgs[i]+'/'

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*9,3), np.float32) # Tablero xadrez 6x9 (H,V)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob(pth+'*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
            print(fname)

        cv2.destroyAllWindows() #::: Testar omitindo esta linha
        
Calibra a câmera

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        
Rectifica as imagens usando a calibração da câmera anterior

    img = cv2.imread(images[1])
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    imgtorectify = glob.glob(pthtorectify+'*.jpg')
        
    pthout = pthtorectify + 'Undistorted_images/'

    if not os.path.exists(pthout):
        os.mkdir(pthout)

    for val in imgtorectify:
        img0 = cv2.imread(val)
        # undistort
        dst = cv2.undistort(img0, mtx, dist, None, newcameramtx)
        # Guarda as imagens rectificadas/calibradas
        cv2.imwrite(pthout+val[len(pthtorectify):-4]+'.jpg', dst)
        print(pthout + val[len(pthtorectify):-4] + '.jpg')

    print("Calibração completada com sucesso")
    
    
