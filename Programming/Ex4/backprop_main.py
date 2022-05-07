from itertools import product as combinations
from tabnanny import verbose
import numpy as np
import backprop_data as data
import backprop_network as network
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


class BestResult:
    def __init__(self) -> None:
        self.batch_size    : int    = 0
        self.accuracy      : float  = 0.0
        self.learning_rate : int    = 0
    
    def update(self,accuracy:float,*, batch_size:int=0,learning_rate:float=0.0,verbose=False )->None:
        if self.accuracy < accuracy:
            self.accuracy = accuracy
            if batch_size:
                if verbose:
                    print(f'\t~ Updated Best {batch_size=}')
                self.batch_size = batch_size
            if learning_rate:
                if verbose:
                    print(f'\t ~ Updated Best {learning_rate=}')
                self.learning_rate = learning_rate
        
    def __str__(self) -> str:
        return f'Best Accuracy is {self.accuracy*100}% when {self.batch_size=}, {self.learning_rate=}'
    
    def __repr__(self)->str:
        return f'BestResult({self.accuracy=}, {self.batch_size=},{self.learning_rate=})'

def section_b(epoch=30):
    def make_plot(i,title,x,res,rate):
        print (f'Generated {title} for {rate}')
        plt.plot(x,res,label=f'{rate}')
        plt.title(title)
        plt.legend()
        plt.xlabel("Epoch")

    print('\tRunning Section B:')
    rates = [0.001, 0.01, 0.1, 1, 10, 100]
    x_axis = np.arange(epoch)
    net = network.Network([784, 40, 10])
    training_data, test_data =  data.load(train_size=10000, test_size=5000)
    results = list(zip(
        *[   
            
            net.SGD(training_data, epochs=epoch, mini_batch_size=10, learning_rate=r, test_data=test_data)
            for r in rates
        ]
    )) # order of results type = train_accuracy,train_loss,test_accuracy

    titles = ['Train Accuracy', 'Train Loss', 'Test Accuracy']
    for i,result_type in enumerate(results):
        for res,rate in zip(result_type,rates):
            make_plot(i,titles[i],x_axis,res,rate)
        plt.show()


def section_c():
    print('\tRunning Section C:')
    training_data, test_data = data.load(train_size=50000,test_size=10000)
    net = network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1,test_data=test_data,quick=True)



def section_d(epoch=30):
    best = BestResult()
    rates = [0.01*i for i in range(1,25)]
    batch_sizes = [i for i in range(1, 25)]
    training_data, test_data =  data.load(train_size=10000, test_size=5000)
    net = network.Network([784, 40, 10])

    print('\tRunning Section D:')
    print(f'\tScanning {batch_sizes=}')
    results = list(zip(
        *[   
            
            net.SGD(training_data, epochs=epoch, mini_batch_size=s, learning_rate=0.1, test_data=test_data,quick=True,silent=True)
            for s in batch_sizes
        ]
    )) # order of results type = train_accuracy,train_loss,test_accuracy  
    test_accuracies = results[-1]
    for s,acc in zip(batch_sizes,test_accuracies):
        best.update(acc[-1], batch_size=s,learning_rate=0.1, verbose=True)
    acc = [acc[-1] for acc in test_accuracies]
    plt.plot(batch_sizes,acc)
    plt.title("Test Accuracy Vs. Mini-Batch Size")
    plt.xlabel("Batch Size")
    plt.show()


    print(f'\tScanning {rates=}')
    results = list(zip(
        *[   
            
            net.SGD(training_data, epochs=epoch, mini_batch_size=best.batch_size, learning_rate=r, test_data=test_data,quick=True,silent=True)
            for r in rates
        ]
    )) # order of results type = train_accuracy,train_loss,test_accuracy    
    test_accuracies = results[-1]
    for rate,acc in zip(rates,test_accuracies):      
        best.update(acc[-1], learning_rate=rate, verbose=True)
    
    acc = [acc[-1] for acc in test_accuracies]
    plt.plot(rates,acc)
    plt.title(f"Test Accuracy Vs. Learning Rate with {best.batch_size=}")
    plt.xlabel("Learning rate")
    plt.show()
    print(best)
    
    
 
def section_d_permutations(epoch=30):
    best = BestResult()
    rates = [0.01*i for i in range(3,15)]
    batch_sizes = [i for i in range(1, 15)]
    training_data, test_data =  data.load(train_size=10000, test_size=5000)
    permutations = combinations(batch_sizes,rates)
    net = network.Network([784, 40, 10])

    print('\tRunning Section D:')
    print(f'\tScanning All Permutations of:')
    print(f'\t  {batch_sizes=}')
    print(f'\t  {rates=}')
    results = list(zip(
        *[   
            
            net.SGD(training_data, epochs=epoch, mini_batch_size=size, learning_rate=learning_rate, test_data=test_data,quick=True,silent=True)
            for size,learning_rate in permutations
        ]
    )) # order of results type = train_accuracy,train_loss,test_accuracy  
    test_accuracies = results[-1]
    for (size,rate),acc in zip(permutations,test_accuracies):
        best.update(acc[-1], batch_size=size,learning_rate=rate, verbose=True)
    # acc = [acc[-1] for acc in test_accuracies]
    # plt.plot(batch_sizes,acc)
    # plt.title("Test Accuracy Vs. Mini-Batch Size")
    # plt.xlabel("Batch Size")
    # plt.show()


    print(f'\tScanning {rates=}')
    results = list(zip(
        *[   
            
            net.SGD(training_data, epochs=epoch, mini_batch_size=best.batch_size, learning_rate=r, test_data=test_data,quick=True,silent=True)
            for r in rates
        ]
    )) # order of results type = train_accuracy,train_loss,test_accuracy    
    test_accuracies = results[-1]
    for rate,acc in zip(rates,test_accuracies):      
        best.update(acc[-1], learning_rate=rate, verbose=True)
    
    acc = [acc[-1] for acc in test_accuracies]
    plt.plot(rates,acc)
    plt.title(f"Test Accuracy Vs. Learning Rate with {best.batch_size=}")
    plt.xlabel("Learning rate")
    plt.show()
    print(best)
   

def main():
    print('Intro to ML Ex4 2022 Q2: Neural Networks ')
    section_b()
    section_c()
    section_d()
    section_d_permutations()



if __name__ == "__main__":
    main()
