import numpy as np


class PcPie:
    @staticmethod
    def pcacool(M,N):
        
        k=z=100
        subject_images = M
        n_components = N

        mean_cols = np.mean(M, axis=0)
        M_meancentered = M - mean_cols

        M_cov = np.cov(M_meancentered)

        eigenValues,eigenVectors = np.linalg.eig(M_cov)
        
        ev = eigenVectors
        idx = np.argsort(np.abs(eigenValues))[::-1]

        
        eigenVectors = eigenVectors[:,idx]
        eigenVectors_top_N = eigenVectors[:,idx[:N]]


        eigenfaces = np.dot(eigenVectors_top_N.T, M_meancentered).reshape(n_components, k, z)
        f_m = np.mean(subject_images, axis=0)
        f_m = np.reshape(f_m, (100,100))


        # weights
        weights = []

        for j in range(n_components):
            img_vector = subject_images[j]
            img_vector = img_vector - f_m.flatten()
            img_vector = np.array(img_vector)
            img_vector = np.reshape(img_vector, (100,100))
            weight = np.dot(img_vector, eigenfaces[j])
            weights.append(weight) 

        return weights, eigenfaces, f_m
    
    @staticmethod
    def assemble(f_m, eigenfaces,weights,num_components, wrong=False):
        
        if(wrong):
            k=z=100

            output=0
            f_m = np.array(f_m)
            f_m = np.reshape(f_m,(100,100))
        
            # multiply all entries of weights with -1 ,
            #weights = np.multiply(weights, -0.6)
            weights = np.multiply(weights, -0.2)

            weights = np.random.randint(30, 31, size=weights.shape)
            #weights = np.multiply(weights, 100)

            # print('good weights: ', weights)
            for i in range(num_components):
           
                output += output + eigenfaces[i] * weights[i][0]

            output = f_m + output
       
            return output
        else:
            k=z=100

            output=0
            f_m = np.array(f_m)
            f_m = np.reshape(f_m,(100,100))
            
            # print('good weights: ', weights)
            for i in range(num_components):
            
                output += output + eigenfaces[i] * weights[i][0]

            output = f_m + output
        
            return output