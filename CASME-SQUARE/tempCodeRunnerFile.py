
# def project_perturbation(data_point,p,perturbation ):
#     if p == 2:
#         perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
#     elif p == np.inf:
#         perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
#     return perturbation

# delta = 0.2
# max_iter_uni=20
# num_classes = 10
# overshoot=0.2
# max_iter_df=20
# xi=10
# p=np.inf
# fooling_rate = 0.0
# iter = 0
# v=np.zeros([64,64])
# net = mdl.ConvNet()
# #validation_labels = numpy.argmax(validation_labels, axis=1)
# transformer = transforms.ToTensor()
# train_images2, validation_images2, train_labels2, validation_labels2 =  train_test_split(training_set2, traininglabels, test_size=0.5, random_state=2)
# validation_labels2 = numpy.argmax(validation_labels2, axis=1)

# while fooling_rate < 1-delta and iter < max_iter_uni:
#         print("Iteration  ", iter)
#         predictions = model.predict(validation_images)
#         predictions_labels = numpy.argmax(predictions, axis=1)


#         path1 = os.path.join(ori_path,str(iter)+"/")    
#         os.mkdir(path1)

#         path3 = os.path.join(pert_path,str(iter)+"/")    
#         os.mkdir(path3)
#         for index in range (len(validation_labels)):
#                 v = v.reshape((v.shape[0], -1))

#                 # Feeding the original image to the network and storing the label returned
#                 r2 = validation_labels[index]

#                 # # Generating a perturbed image from the current perturbation v and the original image
#                 # per_img = Image.fromarray(transformer2(cur_img)+v.astype(np.uint8))
#                 # per_img1 = transformer1(transformer2(per_img))[np.newaxis, :].to(device)

#                 # Feeding the perturbed image to the network and storing the label returned
#                 r1 = predictions_labels[index]

#                 # If the label of both images is the same, the perturbation v needs to be updated
#                 if r1 == r2:
#                         print(">> k =", index, ', pass #', iter, end='      ')

#                 # Finding a new minimal perturbation with deepfool to fool the network on this image
#                 img_tensor = torch.tensor(validation_images2[index])

#                 path2 = os.path.join(path1+str(index)+"/")
#                 os.mkdir(path2)
#                 for i in range(96):
#                         name = str(iter)+"/"+str(index)+"/"+str(i)+".jpg"
#                         frame_img = img_tensor[:, :, :, i]
                        
#                         torchvision.utils.save_image(frame_img, ori_path+name, normalize = True)

#                         dr, iter_k, label, k_i, pert_image = deepfool.deepfool(frame_img, net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

#                         # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
#                         if iter_k < max_iter_df-1:

#                                 # framing = imagenet.ImageNet.get_framing(1)
#                                 # input_att, _ = framing(input=img_tensor)
#                                 # with_frame = input_att.numpy()
#                                 # with_frame = cv2.resize(with_frame, (image_rows, image_columns), interpolation = cv2.INTER_AREA)

#                                 v[:, :] += dr[0,0, :, :]

#                                 v = project_perturbation( xi, p,v)

#         name = str(iter)+".jpg"
#         torchvision.utils.save_image(transformer(v), v_path+name, normalize = True)
#         val_imgs = validation_images2.copy()
#         for i in range (len(validation_images2)):
#                 path4 = os.path.join(path3,str(i)+"/")    
#                 os.mkdir(path4)
#                 val_img = validation_images2[i]
#                 val_img_tensor = torch.tensor(validation_images2[i])
#                 for j in range(96):
#                         name = str(iter)+"/"+str(i)+"/"+str(j)+".jpg"
#                         frame_img_tensor = val_img_tensor[:, :, :, i]
#                         frame_img_tensor += transformer(v).float()

#                         frame_img = val_img[:, :, :, i]
#                         frame_img += v

#                         torchvision.utils.save_image(frame_img_tensor, pert_path+name, normalize = True)
#                         val_imgs[i][:, :, :, i] = frame_img       
#         val_imgs.resize(48,64,64,96)        
#         predictions = model.predict(val_imgs)
#         predictions_labels = numpy.argmax(predictions, axis=1)
#         cfm = confusion_matrix(validation_labels2, predictions_labels)
#         print(cfm)
#         if len(cfm)==3: fooling_rate = (cfm[0][0]+cfm[1][1]+cfm[2][2])/48
#         else: fooling_rate = (cfm[0][0]+cfm[1][1])/48
#         print(fooling_rate)
#         iter = iter + 1

# end = time.time()

# print(" ============ Time taken :",end=" ")
# print(end-start,end =" ===========\n")
