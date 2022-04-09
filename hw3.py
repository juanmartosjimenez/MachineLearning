import numpy as np
from scipy.stats import multivariate_normal
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pandas import get_dummies
import warnings
from scipy.stats import multivariate_normal
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import tensorflow
from sklearn.model_selection import KFold

warnings.filterwarnings("error")


def train_data(x, labels, x_test, labels_test):
    activation_function = 'alu'
    num_perceptrons = 15
    # run the model for 1 to 15 perceptrons
    labels = labels-1
    labels_test = labels_test-1
    y_train = get_dummies(labels[:,0])
    train_len = len(labels_test)
    test_len = len(labels_test)
    start_time = time.process_time()
    all_percp = {}
    for ii in range(1,100):
        model = keras.models.Sequential([keras.layers.Dense(ii, input_dim=3, activation='selu'),
                                         keras.layers.Dense(4, activation=keras.activations.softmax)])

        model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.fit(x, y_train, batch_size = 10, epochs =50, verbose=0)
        predict_x = model.predict(x_test)
        classes_x = np.argmax(predict_x, axis=1)
        print(labels_test[:,0])
        print(classes_x)
        cm = confusion_matrix(labels_test, classes_x, labels=[0,1,2,3])
        print(1-np.trace(cm)/test_len)
        all_percp[ii] = 1-np.trace(cm)/test_len

    print(all_percp)
    print("1-16 perceptrons done for " + str(train_len) + " samples")
    print("Time taken " + str(time.process_time()-start_time))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(list(all_percp.keys()), list(all_percp.values()))
    ax.set_xlabel("Number of perceptrons")
    ax.set_ylabel("Percent error rate")
    ax.set_title("Error rate for " + str(train_len) + " samples")
    plt.savefig("perceptron_vs_error_"+str(train_len)+"_samples")
    exit()

def problem1():
    samples = [100, 200, 500, 1000, 2000, 5000]
    test = 100000
    gen_data = [100, 200, 500, 1000, 2000, 5000, test]
    # keeps track of different covariances for each class
    all_samples = []
    covMatrices = np.zeros((3, 3, 4))
    # Black
    covMatrices[:, :, 0] = np.array([[1.00, 0.01, 0.01], [0.01, 1, 0.01], [0.01, 0.01, 1.00]])
    # Blue
    covMatrices[:, :, 1] = np.array([[1.05, 0.02, 0.01], [0.02, 1.05, 0.02], [0.01, 0.02, 1.00]])
    # Red
    covMatrices[:, :, 2] = np.array([[1.08, 0.04, 0.01], [0.04, 1.05, 0.02], [0.01, 0.02, 1.02]])
    # Green
    covMatrices[:, :, 3] = np.array([[1, 0.07, 0.09], [0.07, 1, 0.01], [0.09, 0.01, 1.00]])
    meanVectors = np.array([[1.5, 1.5, 0], [1.5, 0, 1.5], [0, 1.5, 1.5], [1.5, 1.5, 1.5]])
    priors = np.array([[0.25, 0.25, 0.25, 0.25]])
    # number of classes
    C = 4
    # dimensionality of the data
    n = 3

    # generating data
    test_x = None
    test_labels = None
    for sample in gen_data:

        u = np.random.random((1, sample))
        labels = np.zeros((1, sample))
        x = np.zeros((n, sample))
        thresholds = np.zeros((1, C + 1))
        thresholds[:, 0:C] = np.cumsum(priors)
        thresholds[:, C] = 1
        for l in range(C):
            indl = np.where(u <= float(thresholds[:, l]))
            Nl = len(indl[1])
            labels[indl] = (l + 1) * 1
            u[indl] = 1.1
            try:
                x[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[l, :], covMatrices[:, :, l], Nl))
            except RuntimeWarning as e:
                print(e)
                exit()
        if sample == 100000:
            test_x = x
            test_labels = labels
        else:
            all_samples.append((x, labels))

    # function to plot the data
    def plot3(x, labels):
        sample_num = len(labels)
        mark = "o"
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        c1 = np.where(1 == labels)[1]
        c2 = np.where(2 == labels)[1]
        c3 = np.where(3 == labels)[1]
        c4 = np.where(4 == labels)[1]
        ax.scatter(x[0, c1], x[1, c1], x[2, c1], marker=mark, color="black")
        ax.scatter(x[0, c2], x[1, c2], x[2, c2], marker=mark, color="blue")
        ax.scatter(x[0, c3], x[1, c3], x[2, c3], marker=mark, color="red")
        ax.scatter(x[0, c4], x[1, c4], x[2, c4], marker=mark, color="green")
        plt.legend(["class 1", "class 2", "class 3", "class 4"])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title('Training Dataset ' + str(sample_num) + " samples")
        plt.show()

    # plotting the data
    for ii, num_samples in enumerate(samples):
        # plot3(all_samples[ii][0], all_samples[ii][1], num_samples)
        pass

    # Theoretical Optimal Classifier using test data

    # loss matrix
    loss = np.ones((C, C))
    np.fill_diagonal(loss, 0)

    # calculating optimal probability of error
    N = test
    # p(x|L=1)
    pxgivenl = np.zeros((C, N))

    for ii in range(C):
        pxgivenl[ii, :] = multivariate_normal.pdf(test_x.T, meanVectors[ii, :], covMatrices[:, :, ii])
        # pxgivenl[ii-1] = len(np.where(test_labels == ii)[1])/test

    priors = np.ones((1, C)) / 4
    px = np.matmul(priors, pxgivenl)  # P(B) Total prob theorem
    posteriors = (np.divide(np.multiply(pxgivenl, np.tile(priors.T, (1, test))), px))
    expectedRisks = np.matmul(loss, posteriors)
    decisions = np.argmin(expectedRisks, axis=0)
    minPfe = np.sum(np.logical_xor(decisions, test_labels - 1)) / test

    # 10-fold cross-validation using minimum classification error probability as the objective function

    for x, label in all_samples:
        # Number of folds
        k = 10
        kf = KFold(n_splits=k, shuffle=True)
        X = x.T
        L = label.T
        for train_index, test_index in kf.split(x.T):
            x_train = X[train_index]
            x_labels_train = L[train_index]
            x_test = X[test_index]
            x_labels_test = L[test_index]
            train_data(x_train, x_labels_train, x_test, x_labels_test)
            exit()


        exit()


if __name__ == "__main__":
    problem1()
