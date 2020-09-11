import matplotlib.pyplot as plt
fcn_sup = [0.562,0.506,0.503,0.452,0.450,0.417,0.411]
angle = [0,10,350,20,340,30,330]
angle = [-30,-20,-10,0,10,20,30]
angle = [str(i) for i in angle] 
fcn_sup_rot_val = [0.514,0.530,0.541,0.547,0.542,0.530,0.513]  
fcn_multi_task_train = [0.401,0.418,0.471,0.526,0.468,0.418,0.400]
fcn_multi_task_val = [0.325,0.330,0.370,0.421,0.372,0.333,0.325]
fcn_sup_train = [0.352,0.389,0.460,0.524,0.457,0.374,0.338]
fcn_sup_val = [0.299,0.322,0.375,0.427,0.373,0.317,0.292]
plt.figure()
plt.title('Evolution of the IoU VAL VOC 2012 according to the rotation of the input images.')
plt.plot(angle,fcn_sup_val,label='fcn_supervised')
plt.plot(angle,fcn_multi_task_val,label='fcn loss multi task equiv')
plt.plot(angle,fcn_sup_rot_val,label='fcn_sup rotation data aug')
plt.xlabel("input image rotation angle")
plt.ylabel("Mean IoU")
plt.legend(loc="upper right")
plt.show()
plt.figure()
plt.title('Evolution of the IoU TRAIN VOC 2012 according to the rotation of the input images.')
plt.plot(angle,fcn_sup_train,label='fcn_supervised')
plt.plot(angle,fcn_multi_task_train,label='fcn loss multi task equiv')
plt.xlabel("input image rotation angle")
plt.ylabel("Mean IoU")
plt.legend(loc="upper right")
plt.show()