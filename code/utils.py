import numpy as np

def checker( a, b, n=50 ):        
    assert a.shape == b.shape
    w,h = a.shape        
    for d in range( n ):    
        r = np.zeros( n )        
        if d==0:            
            r[:n//2]=1
            r1=r
        elif d<n//2:
            r[:n//2]=1
            r1=np.vstack( (r1,r) )
        else:
            r[n//2:]=1    
            r1=np.vstack( (r1,r) )
            
    ar = np.array( r1 )    
    
    W,H=ar.shape     
    W=w//W
    H=h//H
    x0 = np.tile( ar, [W,H])    
    x = np.pad( x0, ((0,0),(w - x0.shape[0], h - x0.shape[1])) )    
    
    res = a*(1 - x) + b*x
    plt.imshow(res)    
    return res, x0, x, r1

res, x0, x, r1 = checker( rescale(img[:,:,1]), label )