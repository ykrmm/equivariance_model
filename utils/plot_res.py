import matplotlib.pyplot as plt
#fcn_sup = [0.562,0.506,0.503,0.452,0.450,0.417,0.411]
angle = [-30,-20,-10,0,10,20,30]
angle = [str(i) for i in angle] 
#fcn_sup_rot_val = [0.514,0.530,0.541,0.547,0.542,0.530,0.513]  
fcn_multi_task_train = [0.401,0.418,0.471,0.526,0.468,0.418,0.400]
fcn_multi_task_val = [0.325,0.330,0.370,0.421,0.372,0.333,0.325]
fcn_rot_sup_train = [0.509,0.533,0.546,0.550,0.545,0.534,0.513]
fcn_rot_sup_val = [0.420,0.435,0.446,0.454,0.446,0.434,0.420]
fcn_sup_train = [0.352,0.389,0.460,0.524,0.457,0.374,0.338]
fcn_sup_val = [0.299,0.322,0.375,0.427,0.373,0.317,0.292]
fcn_semisup_train = [0.461,0.482,0.531,0.582,0.533,0.476,0.452]
fcn_semisup_val = [0.384,0.396,0.431,0.472,0.432,0.391,0.375]
fcn_semisup_train_KL = [0.466,0.470,0.506,0.539,0.505,0.465,0.460]
fcn_semisup_val_KL = [0.448,0.447,0.481,0.513,0.480,0.442,0.428]

plt.figure()
plt.title('Evolution of the IoU VAL VOC 2012 according to the rotation of the input images.')
plt.plot(angle,fcn_sup_val,label='fcn_supervised')
plt.plot(angle,fcn_multi_task_val,label='fcn with multi task loss equiv CE')
plt.plot(angle,fcn_semisup_val,label='fcn with semi sup loss equiv CE')
plt.plot(angle,fcn_semisup_val_KL,label='fcn with semi sup loss equiv KL')
plt.plot(angle,fcn_rot_sup_val,label='fcn rot* Random rotation for data augmentation')
#plt.plot(angle,fcn_sup_rot_val,label='fcn_sup rotation data aug')
plt.xlabel("input image rotation angle")
plt.ylabel("Mean IoU")
plt.legend(loc="upper right")
plt.show()
plt.figure()
plt.title('Evolution of the IoU TRAIN VOC 2012 according to the rotation of the input images.')
plt.plot(angle,fcn_sup_train,label='fcn_supervised')
plt.plot(angle,fcn_multi_task_train,label='fcn loss multi task equiv CE')
plt.plot(angle,fcn_semisup_train,label='fcn with semi sup loss equiv CE')
plt.plot(angle,fcn_semisup_train_KL,label='fcn with semi sup loss equiv KL')
plt.plot(angle,fcn_rot_sup_train,label='fcn rot* Random rotation for data augmentation')
plt.xlabel("input image rotation angle")
plt.ylabel("Mean IoU")
plt.legend(loc="upper right")
plt.show()