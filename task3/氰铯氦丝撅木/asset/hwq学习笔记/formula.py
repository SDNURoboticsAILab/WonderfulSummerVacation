import numpy as np 
import math

def Rx(a) :
    ans = np.array([[1,0,0],[0,math.cos(a),-1*math.sin(a)],[0,math.sin(a),math.cos(a)]])
    return ans 

def Ry(a) :
    ans = np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-1*math.sin(a),0,math.cos(a)]])
    return ans 
    
def Rz(a) :
    ans = np.array([[math.cos(a), -1*math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
    return ans


    
    