import subprocess
import numpy as np

for i in range(20,21):
	for j in range(20,21):
		print('Going to train with states = {} and symbols = {}'.format(i,j))
		subprocess.call(["./train_hmm", "train_class_seq1.txt", "1234", str(i), str(j), ".01"])
		subprocess.call(["./train_hmm", "train_class_seq2.txt", "1234", str(i), str(j), ".01"])
		subprocess.call(["./train_hmm", "train_class_seq3.txt", "1234", str(i), str(j), ".01"])



		#Predict
		print('Testing on train data class1 now')
		subprocess.call(["./test_hmm", "train_class_seq1.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "train_class_seq1.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "train_class_seq1.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		count = 0
		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			#print('printing for fun {}'.format(alphas_class1[i]))
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 0
			if (assigned_class_label == org_class_label):
				count+=1

		n_train_class1 = len(alphas_class1)		


		alphas_class1 = []
		alphas_class2 = []
		alphas_class3 = []
	
		subprocess.call(["./test_hmm", "train_class_seq2.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "train_class_seq2.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "train_class_seq2.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 1
			if (assigned_class_label == org_class_label):
				count+=1

		n_train_class2 = len(alphas_class1)		

		alphas_class1 = []
		alphas_class2 = []
		alphas_class3 = []
	
		subprocess.call(["./test_hmm", "train_class_seq3.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "train_class_seq3.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "train_class_seq3.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 2
			if (assigned_class_label == org_class_label):
				count+=1	

		n_train_class3 = len(alphas_class1)	

		train_accuracy = count/(n_train_class1 + n_train_class2 + n_train_class3)*100
		
		#print('total samples: {}'.format(n_train_class1 + n_train_class2 + n_train_class3))
		
		#for val accuracy
		#Predict
		alphas_class1 = []
		alphas_class2 = []
		alphas_class3 = []
		print('Testing on val data class1 now')
		subprocess.call(["./test_hmm", "val_class_seq1.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "val_class_seq1.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "val_class_seq1.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		count = 0
		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			#print('printing for fun {}'.format(alphas_class1[i]))
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 0
			if (assigned_class_label == org_class_label):
				count+=1

		n_train_class1 = len(alphas_class1)		


		alphas_class1 = []
		alphas_class2 = []
		alphas_class3 = []
	
		subprocess.call(["./test_hmm", "val_class_seq2.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "val_class_seq2.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "val_class_seq2.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 1
			if (assigned_class_label == org_class_label):
				count+=1

		n_train_class2 = len(alphas_class1)		

		alphas_class1 = []
		alphas_class2 = []
		alphas_class3 = []
	
		subprocess.call(["./test_hmm", "val_class_seq3.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "val_class_seq3.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "val_class_seq3.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 2
			if (assigned_class_label == org_class_label):
				count+=1	

		n_train_class3 = len(alphas_class1)	

		val_accuracy = count/(n_train_class1 + n_train_class2 + n_train_class3)*100

		#for test accuracy
		#Predict
		alphas_class1 = []
		alphas_class2 = []
		alphas_class3 = []
		print('Testing on test data class1 now')
		subprocess.call(["./test_hmm", "test_class_seq1.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "test_class_seq1.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "test_class_seq1.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		count = 0
		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			#print('printing for fun {}'.format(alphas_class1[i]))
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 0
			if (assigned_class_label == org_class_label):
				count+=1

		n_train_class1 = len(alphas_class1)		


		alphas_class1 = []
		alphas_class2 = []
		alphas_class3 = []
	
		subprocess.call(["./test_hmm", "test_class_seq2.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "test_class_seq2.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "test_class_seq2.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 1
			if (assigned_class_label == org_class_label):
				count+=1

		n_train_class2 = len(alphas_class1)		

		alphas_class1 = []
		alphas_class2 = []
		alphas_class3 = []
	
		subprocess.call(["./test_hmm", "test_class_seq3.txt", "train_class_seq1.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class1_list = file.readlines()
		alphas_class1 = alphas_class1_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class1')
		#print(alphas_class1)

		subprocess.call(["./test_hmm", "test_class_seq3.txt", "train_class_seq2.txt.hmm"])
		file = open("alphaout","r")
		alphas_class2_list = file.readlines()
		alphas_class2 = alphas_class2_list[0].split(" ")
		file.close()
		#print('going to print alpha values for class2')
		#print(alphas_class2)

		subprocess.call(["./test_hmm", "test_class_seq3.txt", "train_class_seq3.txt.hmm"])
		file = open("alphaout","r") 
		alphas_class3_list = file.readlines()
		alphas_class3 = alphas_class3_list[0].split(" ")
		#print('going to print alpha values for class3')
		#print(alphas_class3)
		file.close()

		total_samples = len(alphas_class1)
		for i in range(1,total_samples):
			assigned_class_label = np.argmax([float(alphas_class1[i]), float(alphas_class2[i]), float(alphas_class3[i])])
			org_class_label = 2
			if (assigned_class_label == org_class_label):
				count+=1	

		n_train_class3 = len(alphas_class1)	

		test_accuracy = count/(n_train_class1 + n_train_class2 + n_train_class3)*100
		print('Train Accuracy is {}'.format(train_accuracy))
		print('Val Accuracy is {}'.format(val_accuracy))
		print('Test Accuracy is {}'.format(test_accuracy))
		#print('total samples: {}'.format(n_train_class1 + n_train_class2 + n_train_class3))
		

		
		
 



#train_class_seq3.txt 1234 2 4 .01