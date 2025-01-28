import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def visualize_data(df: pd.DataFrame) -> None:
    """Plot year vs. number of frozen days and save it to "data_plot.jpg" using plt.savefig.

    Args:
        df (pd.DataFrame): pd.DataFrame of the csv file
    """
    plt.figure()
    plt.plot(df["year"], df["days"])
    plt.title('year vs. number of frozen days')
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.savefig("data_plot.jpg")

def normalization_vector(df: pd.DataFrame) -> np.ndarray:
    """Print the normalized and augmented data matrix X

    Args:
        df (pd.DataFrame): pd.DataFrame of the csv file
    """
    x = np.array(df["year"])
    x_normalized = (x - x.min()) / (x.max() - x.min())
    X = np.vstack((x_normalized, np.ones(x_normalized.shape[0])))
    X_normalized = np.transpose(X) # (x, 1), shape (n, 2)
    print("Q3:")
    print(X_normalized)
    return X_normalized

def closed_form_solution(df: pd.DataFrame, X_normalized: np.ndarray) -> np.ndarray:
    """Print the optimal weight and bias as a numpy array

    Args:
        df (pd.DataFrame): pd.DataFrame of the csv file
        X_normalized (np.ndarray): result of normalization_vector()
    """
    Y = np.array(df["days"])
    weights = np.linalg.inv(np.transpose(X_normalized) @ X_normalized) @ np.transpose(X_normalized) @ Y
    print("Q4:")
    print(weights)
    return weights

def linear_regression_with_gradient_descent(df: pd.DataFrame, X_normalized: np.ndarray, learning_rate: float, iterations: int):
    """Imitate gradient descent

    Args:
        df (pd.DataFrame): pd.DataFrame of the csv file
        X_normalized (np.ndarray): result of normalization_vector(), shape(n, 2)
        learning_rate: learning_rate from sys
        iterations: iterations from sys
    """
    Y = np.array(df["days"]) # shape(n, 1)
    weights_init = np.zeros((2, 1)).astype("float64") # initial weights 0, 0, shape(2, 1)
    weights = weights_init
    loss_record = []
    print("Q5a:")
    for iter in range(iterations):
        loss = np.linalg.norm(X_normalized @ weights - Y, 2) / 2 / Y.shape[0]
        loss_record.append(loss)
        if iter % 10 == 0:
            print(np.transpose(weights)[0])
        y_hat = np.transpose(weights) @ np.transpose(X_normalized) # shape(1, n)
        gradient = np.transpose((y_hat - np.transpose(Y)) @ X_normalized / Y.shape[0]) # shape(2, 1)
        weights -= learning_rate * gradient

    print("Q5b: 0.3")
    print("Q5c: 450")

    iteration = [_ for _ in range(iterations)]
    plt.figure()
    plt.plot(iteration, loss_record)
    plt.title("loss_plot")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg")

def prediction(df: pd.DataFrame, weights: np.ndarray) -> None:
    """Print the model’s prediction for the number of ice days for 2023-24

    Args:
        df (pd.DataFrame): pd.DataFrame of the csv file
        weights (np.ndarray): weights from closed_form_solution
    """
    x = np.array(df["year"])
    x_pre = 2023
    y_hat = (x_pre - x.min()) / (x.max() - x.min()) * weights[0] + weights[1]
    print("Q6: " + str(y_hat))

def model_interpretation(weights: np.ndarray) -> None:
    """Print the sign of w and interpret all three possible signs.

    Args:
        weights (np.ndarray): weights from closed_form_solution
    """
    if weights[0] > 0:
        symbol = ">"
    elif weights[0] < 0:
        symbol = "<"
    else:
        symbol = "="
    print("Q7a: " + symbol)
    print("Q7b: The symbol “>” indicates that as the years increase, the number of ice days is also increasing; the symbol “<” indicates the opposite; the symbol “=” indicates that there is currently no correlation between ice days and years.")

def model_limitations(df: pd.DataFrame, weights: np.ndarray):
    """ Print the model's prediction for the year Lake Mendota will no longer freeze, and write a few sentences analyzing the prediction.

    Args:
        df (pd.DataFrame): pd.DataFrame of the csv file
        weights (np.ndarray): weights from closed_form_solution
    """
    x = np.array(df["year"])
    x_star = -weights[1] * (x.max() - x.min()) / weights[0] + x.min()
    print("Q8a: " + str(x_star))
    print("Q8b: The predictions of a linear regression model outside the data range may be unreliable and can only indicate a general trend within the range. X* is not a particularly convincing prediction, as other factors, such as wind speed, precipitation, and relevant policies, may cause fluctuations in the data that linear regression cannot capture.")

if __name__ == "__main__":
    print(sys.argv)
    file_path = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])
    df = pd.read_csv(file_path, index_col=False)
    visualize_data(df)
    X_normalized = normalization_vector(df)
    weights = closed_form_solution(df, X_normalized)
    linear_regression_with_gradient_descent(df, X_normalized, learning_rate, iterations)
    prediction(df, weights)
    model_interpretation(weights)
    model_limitations(df, weights)

    
