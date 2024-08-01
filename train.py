import numpy as np
import matplotlib.pyplot as plt

global indexs
indexs = 0


class NeuralNetwork:
    def __init__(
        self,
        layers,
        learning_rate=0.1,
        momentum_rate=0.9,
        activation_function="sigmoid",
    ):
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.activation_function = activation_function
        self.w, self.delta_w, self.b, self.delta_b, self.local_gradient = (
            self._init_params()
        )

    def _init_params(self):
        weights = []
        delta_weights = []
        biases = []
        delta_biases = []
        local_gradients = [np.zeros(self.layers[0])]
        for i in range(1, len(self.layers)):
            weight = np.random.randn(self.layers[i], self.layers[i - 1]) * np.sqrt(
                2.0 / self.layers[i - 1]
            )
            weights.append(weight)
            delta_weights.append(np.zeros_like(weight))
            bias = np.zeros(self.layers[i])
            biases.append(bias)
            delta_biases.append(np.zeros_like(bias))
            local_gradients.append(np.zeros(self.layers[i]))
        return weights, delta_weights, biases, delta_biases, local_gradients

    def _activation(self, x):
        if self.activation_function == "linear":
            return x
        elif self.activation_function == "relu":
            return np.maximum(0, x)
        elif self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)

    def _activation_diff(self, x):
        if self.activation_function == "linear":
            return np.ones_like(x)
        elif self.activation_function == "relu":
            return np.where(x > 0, 1.0, 0.0)
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "tanh":
            return 1 - x**2

    def _forward(self, input_data):
        self.V = [input_data]
        for i in range(len(self.layers) - 1):
            self.V.append(self._activation(np.dot(self.V[i], self.w[i].T) + self.b[i]))

    def _backward(self, design_output):
        error = design_output - self.V[-1]
        for i in reversed(range(len(self.layers) - 1)):
            if i == len(self.layers) - 2:
                self.local_gradient[i + 1] = error * self._activation_diff(
                    self.V[i + 1]
                )
            else:
                self.local_gradient[i + 1] = self._activation_diff(
                    self.V[i + 1]
                ) * np.dot(self.local_gradient[i + 2], self.w[i + 1])
            self.delta_w[i] = (self.momentum_rate * self.delta_w[i]) + np.outer(
                self.local_gradient[i + 1], self.V[i]
            )
            self.delta_b[i] = (
                self.momentum_rate * self.delta_b[i]
            ) + self.local_gradient[i + 1]
            self.w[i] += self.learning_rate * self.delta_w[i]
            self.b[i] += self.learning_rate * self.delta_b[i]
        return np.sum(error**2) / 2

    def train(self, input_data, design_output, epochs=1000, tol=0.001):
        errors = []
        for epoch in range(epochs):
            epoch_error = 0
            for i in range(len(input_data)):
                self._forward(input_data[i])
                epoch_error += self._backward(design_output[i])
            avg_error = epoch_error / len(input_data)
            errors.append(avg_error)
            if avg_error < tol:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            if epoch % 100 == 0:
                print(f"Epoch = {epoch + 1} / {epochs} | Avg Error = {avg_error}")
        return errors

    def test(self, input_data, design_output, task_type="classification"):
        global indexs
        actual_output = []
        for i in input_data:
            self._forward(i)
            actual_output.append(self.V[-1])
        if task_type == "classification":
            actual_output = [np.argmax(output) for output in actual_output]
            design_output = [np.argmax(output) for output in design_output]
            accuracy = np.mean(np.array(actual_output) == np.array(design_output)) * 100
            print(f"Classification Accuracy = {accuracy:.2f}%")

            # num_classes = max(design_output) + 1
            # conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
            # for true_label, pred_label in zip(design_output, actual_output):
            #     conf_matrix[true_label, pred_label] += 1
            # plt.figure(figsize=(10, 7))
            # plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
            # plt.title("Confusion Matrix")
            # plt.colorbar()
            # classes = np.arange(num_classes)
            # plt.xticks(classes, classes)
            # plt.yticks(classes, classes)
            # for i in range(num_classes):
            #     for j in range(num_classes):
            #         plt.text(
            #             j, i, conf_matrix[i, j], ha="center", va="center", color="black"
            #         )

            # plt.xlabel("Predicted Label")
            # plt.ylabel("True Label")
            # print(indexs)
            # plt.savefig(f"classification_F{indexs}.png")
            # indexs += 1
            # plt.show()

            return accuracy
        else:
            actual_output = [output[0] for output in actual_output]
            error = np.mean(np.abs(np.array(actual_output) - np.array(design_output)))
            print(f"Regression Mean Absolute Error = {error:.2f}")

            # plt.figure(figsize=(10, 6))
            # plt.plot(
            #     range(len(actual_output)),
            #     actual_output,
            #     label="Actual Output",
            #     marker="o",
            #     linestyle="-",
            # )
            # plt.plot(
            #     range(len(design_output)),
            #     design_output,
            #     label="Design Output",
            #     marker="x",
            #     linestyle="--",
            # )
            # plt.legend()
            # plt.title("Regression: Actual vs Design Output")
            # plt.xlabel("Sample Index")
            # plt.ylabel("Output Value")
            # print(indexs)
            # plt.savefig(f"regression_F{indexs}.png")
            # indexs += 1
            # plt.show()

            return error


def ReadData(filename="Flood_dataset.txt", data_type="regression"):
    data = []
    input_data = []
    output_data = []
    if data_type == "regression":
        with open(filename) as f:
            lines = f.readlines()[2:]
            data = [list(map(float, line.split())) for line in lines]
        data_np = np.array(data)
        input_raw = data_np[:, :8]
        output_raw = data_np[:, 8]
        input_data = (input_raw - np.min(input_raw, axis=0)) / (
            np.max(input_raw, axis=0) - np.min(input_raw, axis=0)
        )
        output_data = (output_raw - np.min(output_raw)) / (
            np.max(output_raw) - np.min(output_raw)
        )
    elif data_type == "classification":
        with open(filename) as f:
            a = f.readlines()
            for line in range(1, len(a), 3):
                input_data.append(
                    np.array(
                        [float(element) for element in a[line][:-1].split()]
                    ).tolist()
                )
                output_data.append(
                    np.array(
                        [float(element) for element in a[line + 1].split()]
                    ).tolist()
                )
    return input_data, output_data


def KFoldValidation(data, k=10):
    input_data = np.array(data[0])
    output_data = np.array(data[1])
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)
    input_data, output_data = input_data[indices], output_data[indices]
    fold_size = len(input_data) // k
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        test_data = (input_data[start:end], output_data[start:end])
        train_data = (
            np.concatenate([input_data[:start], input_data[end:]], axis=0),
            np.concatenate([output_data[:start], output_data[end:]], axis=0),
        )
        yield test_data, train_data


if __name__ == "__main__":
    filename = {"regression": "Flood_dataset.txt", "classification": "cross.txt"}
    layers = {
        "regression": [8, 16, 1],
        "classification": [2, 16, 2],
    }
    output_accuracy_text = {
        "regression": "Regression Mean Error: ",
        "classification": "Classification Accuracy: ",
    }
    activation_function_arr = ["sigmoid", "relu", "tanh", "linear"]
    data_type_arr = ["regression", "classification"]

    k = 10
    learning_rate = 0.1
    momentum_rate = 0.9
    max_epoch = 1000
    av_error = 0.001
    activation_function = "sigmoid"  # 'sigmoid' or 'relu' or 'tanh' or 'linear'
    data_type = "classification"  # "regression" or "classification"

    if input("Do you want to change the default value? (y/n): ").lower() == "y":
        while True:
            print(f"1. K Fold Validation ({k})")
            print(f"2. Learning Rate ({learning_rate})")
            print(f"3. Momentum Rate ({momentum_rate})")
            print(f"4. Max Epoch ({max_epoch})")
            print(f"5. AV Error ({av_error})")
            print(f"6. Data Type ({data_type})")
            print(f"7. Activation Function ({activation_function})")
            print(f"8. End Change Value")
            index = int(input("Select index to change the default value: "))
            if index == 1:
                input_k = input("Enter K Fold Validation: ")
                k = int(input_k) if input_k != "" else k
            elif index == 2:
                input_learning_rate = input("Enter Learning Rate: ")
                learning_rate = (
                    float(input_learning_rate)
                    if input_learning_rate != ""
                    else learning_rate
                )
            elif index == 3:
                input_momentum_rate = input("Enter Momentum Rate: ")
                momentum_rate = (
                    float(input_momentum_rate)
                    if input_momentum_rate != ""
                    else momentum_rate
                )
            elif index == 4:
                input_max_epoch = input("Enter Max Epoch: ")
                max_epoch = int(input_max_epoch) if input_max_epoch != "" else max_epoch
            elif index == 5:
                input_av_error = input("Enter AV Error: ")
                av_error = float(input_av_error) if input_av_error != "" else av_error
            elif index == 6:
                input_data_type = input("Enter Data Type: ").lower()
                data_type = (
                    input_data_type if input_data_type in data_type_arr else data_type
                )
            elif index == 7:
                input_activation_function = input("Enter Activation Function: ").lower()
                activation_function = (
                    input_activation_function
                    if input_activation_function in activation_function_arr
                    else activation_function
                )
            elif index == 8:
                break
            else:
                print(
                    "Invalid selection. Please enter a number corresponding to the options."
                )

    input_data, output_data = ReadData(
        filename=filename[data_type], data_type=data_type
    )
    scores = []
    for train_data, test_data in KFoldValidation((input_data, output_data), k):
        input_train, output_train = train_data
        input_test, output_test = test_data
        nn = NeuralNetwork(
            layers=layers[data_type],
            learning_rate=learning_rate,
            momentum_rate=momentum_rate,
            activation_function=activation_function,
        )
        nn.train(input_train, output_train, epochs=max_epoch, tol=av_error)
        scores.append(nn.test(input_test, output_test, task_type=data_type))
    output_accuracy = np.mean(scores)
    print("--------------------------------")
    print("Settings:")
    print(f"K Fold Validation: {k}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Momentum Rate: {momentum_rate}")
    print(f"Max Epoch: {max_epoch}")
    print(f"AV Error: {av_error}")
    print(f"Data Type: {data_type}")
    print(f"Activation Function: {activation_function}")
    print("--------------------------------")
    if data_type == "regression":
        print(f"{output_accuracy_text[data_type]} {output_accuracy}")
    elif data_type == "classification":
        print(f"{output_accuracy_text[data_type]} {output_accuracy}%")

    # Plotting the graph
    folds = list(range(1, k + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(folds, scores, marker="o", linestyle="-", color="b")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Scores for each Fold in K-Fold Cross-Validation")
    plt.grid(True)
    plt.savefig(f"{data_type} {output_accuracy}.png")
    # plt.show()
