import numpy as np
import submission as sub
if __name__ == '__main__':
    r=np.vstack([2,2,3])
    # print('original r',r)
    R=sub.rodrigues(r)
    # print('R=',R)
    # R=np.eye(3)
    print('R=',R)
    r_new=sub.invRodrigues(R)
    print('inverse r',r_new)
