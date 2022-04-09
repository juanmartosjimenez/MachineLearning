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
import tensorflow as tf
from sklearn.model_selection import KFold

tf.config.list_physical_devices('GPU')
warnings.filterwarnings("error")


def train_data(train_x, train_labels, test_x, test_labels, model):
    activation_function = 'alu'

    # minus 1 to make labels 0 indexed
    train_labels = train_labels - 1
    test_labels = test_labels - 1
    y_train = get_dummies(train_labels[:, 0])
    test_len = len(test_labels)

    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, y_train, batch_size=32, epochs=10, verbose=0)
    predict_x = model.predict(test_x)
    classes_x = np.argmax(predict_x, axis=1)
    cm = confusion_matrix(test_labels, classes_x, labels=[0, 1, 2, 3])
    error_rate = 1 - np.trace(cm) / test_len
    return error_rate


def problem1():
    samples = [100, 200, 500, 1000, 2000, 5000]
    test = 100000
    gen_data = [100,200,500,1000,2000,5000, test]
    # keeps track of different covariances for each class
    all_samples = {}
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
            all_samples[sample] = (x, labels)

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
    for ii, sample in enumerate(samples):
        # plot3(all_samples[ii][0], all_samples[ii][1])
        pass
    # plot3(test_x,test_labels)

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
    print("Theoretical minimum probability of error", minPfe)

    # 10-fold cross-validation using minimum classification error probability as the objective function
    def k_fold_validation():
        performance_metrics = {}
        for x, label in all_samples.values():

            # iterate over number of perceptrons
            k = 10

            kf = KFold(n_splits=k, shuffle=True)
            X = x.T
            num_samples = len(label[0, :])
            L = label.T
            ii = 0
            performance_metrics[num_samples] = {}
            for percep in range(1, 15):
                model = keras.models.Sequential([keras.layers.Dense(percep, input_dim=3, activation='selu'),
                                                 keras.layers.Dense(4, activation=keras.activations.softmax)])

                start_time = time.process_time()
                total_error_rate = 0
                # 10-fold split for each perceptron layer
                for train_index, test_index in kf.split(x.T):
                    x_train = X[train_index]
                    x_labels_train = L[train_index]
                    x_test = X[test_index]
                    x_labels_test = L[test_index]
                    total_error_rate += train_data(x_train, x_labels_train, x_test, x_labels_test, model)

                average_error_rate = total_error_rate / k
                print(average_error_rate)
                performance_metrics[num_samples][percep] = average_error_rate
                print("Finished perceptron", percep, "for", num_samples, "samples time taken",
                      time.process_time() - start_time)

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(list(performance_metrics[num_samples].keys()), list(performance_metrics[num_samples].values()))
            ax.set_xlabel("Number of perceptrons")
            ax.set_ylabel("Percent error rate")
            ax.set_title("Error rate for " + str(num_samples) + " samples")
            plt.savefig("perceptron_vs_error_" + str(num_samples) + "_samples")

    ideal_num_perceptrons = {100: 30, 200: 20, 500: 15, 1000: 10, 2000: 10, 5000: 5}

    def train_test():

        for key, val in ideal_num_perceptrons.items():
            model = keras.models.Sequential([keras.layers.Dense(val, input_dim=3, activation='selu'),
                                             keras.layers.Dense(4, activation=keras.activations.softmax)])

            x_train, y_train = all_samples[key]
            error_rate = train_data(x_train.T, y_train.T, test_x.T, test_labels.T, model)
            print("error rate", error_rate, "samples",key)
    train_test()

if __name__ == "__main__":
    problem1()
