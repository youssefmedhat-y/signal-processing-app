#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def ReadSignalFile(file_name):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices,expected_samples


# In[ ]:


def SinCosSignalSamplesAreEqual(user_choice,file_name,indices,samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
                
    print(len(expected_samples),len(samples))            
    if len(expected_samples)!=len(samples):
        print(user_choice+" Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print(user_choice+" Test case failed, your signal have different values from the expected one") 
            return
    print(user_choice +" Test case passed successfully")


# In[ ]:


def SubSignalSamplesAreEqual(userFirstSignal,userSecondSignal,Your_indices,Your_samples):
    if(userFirstSignal=='Signal1.txt' and userSecondSignal=='Signal2.txt'):
        file_name="D:\code\python/f1\Lab 1\Task1 files+test\Lab 2\Task2 files + testcases\output_Math_operation_Files\signal1-signal2.txt" # write here path of signal1-signal2
    elif(userFirstSignal=='Signal1.txt' and userSecondSignal=='Signal3.txt'):
        file_name="" # write here path of signal1-signal3
        
    expected_indices,expected_samples=ReadSignalFile(file_name)   
    
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Subtraction Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Subtraction Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Subtraction Test case failed, your signal have different values from the expected one") 
            return
    print("Subtraction Test case passed successfully")


# In[ ]:


def NormalizeSignal(MinRange,MaxRange,Your_indices,Your_samples):
    if(MinRange==-1 and MaxRange==1):
        file_name="D:/code/python/f1/Lab 2/Task2 files + testcases/output_Math_operation_Files/normalize of signal 1 (from -1 to 1)-- output.txt" # write here path of normalize signal 1 output.txt
    elif(MinRange==0 and MaxRange==1):
        file_name="" # write here path of normalize signal 2 output.txt
        
    expected_indices,expected_samples=ReadSignalFile(file_name)      
    
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Normalization Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Normalization Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Normalization Test case failed, your signal have different values from the expected one") 
            return
    print("Normalization Test case passed successfully")


# In[ ]:


# use this twice one for Accumlation and one for Squaring
# Task name when call ACC or SQU


#TaskName => choose it (string explain the name of task like (adding,subtracting, .... etc.))
#output_file_name => output file path (given by TAs)
# Your_indices => your indices list from your code (generated/calculated by you)
# Your_samples => your samples list from your code (generated/calculated by you)
def SignalSamplesAreEqual(TaskName,output_file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
    with open(output_file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print(TaskName+" Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print(TaskName+" Test case failed, your signal have different indicies from the expected one") 
            return             
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print(TaskName+" Test case failed, your signal have different values from the expected one") 
            return
    print(TaskName+" Test case passed successfully")

indecies , samples =ReadSignalFile("D:/code/python/f1/Lab 2/myoutputs/sine_360.0Hz.txt")
cat="Sinusoidal Signal Test Case"
SinCosSignalSamplesAreEqual(cat,"D:\code\python/f1\Lab 2\Task2 files + testcases\Sin_Cos\SinOutput.txt",indecies,samples)

indecies , samples =ReadSignalFile("D:\code\python/f1\Lab 2\myoutputs\cosine.txt")
cat="Cosine Signal Test Case"
SinCosSignalSamplesAreEqual(cat,"D:\code\python/f1\Lab 2\Task2 files + testcases\Sin_Cos\CosOutput.txt",indecies,samples)

indecies , samples =ReadSignalFile("D:\code\python/f1\Lab 2\myoutputs\difference_Signal1_Signal2.txt")
SubSignalSamplesAreEqual('Signal1.txt','Signal2.txt',indecies,samples)

indecies , samples =ReadSignalFile("D:\code\python/f1\Lab 2\myoutputs\Signal1_accumulated.txt")
SignalSamplesAreEqual("Accumulation","D:\code\python/f1\Lab 1\Task1 files+test\Lab 2\Task2 files + testcases\output_Math_operation_Files\output accumulation for signal1.txt",indecies,samples)

indecies , samples =ReadSignalFile("D:\code\python/f1\Lab 2\myoutputs\Signal1_normalized_-1_to_1.txt")
NormalizeSignal(-1,1,indecies,samples)