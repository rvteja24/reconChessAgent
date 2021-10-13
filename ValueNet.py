from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential, Model, models
from tensorflow.keras.losses import binary_crossentropy
from tensorflow import concat, square, reduce_sum
from tensorflow.keras.layers import Dense, Flatten, ReLU, Input, Conv2D, BatchNormalization, Add
from tensorflow.keras.activations import relu, softmax, tanh

class ChessModel(Model):
    def __init__(self, input_dimensions, output_dimensions_policy):
        super(ChessModel, self).__init__()
        self.l2_const = 1e-4
        self.input_dim = input_dimensions
        self.dense1 = Dense(256, activation=relu,kernel_regularizer=l2(self.l2_const), kernel_initializer='glorot_uniform')
        self.dense2 = Dense(256, activation=relu,kernel_regularizer=l2(self.l2_const), kernel_initializer='glorot_uniform')
        self.normalize1 = BatchNormalization()
        self.normalize2 = BatchNormalization()
        self.normalize3 = BatchNormalization()
        self.normalize4 = BatchNormalization()
        self.normalize5 = BatchNormalization()
        self.normalize6 = BatchNormalization()
        self.normalize7 = BatchNormalization()
        self.normalize8 = BatchNormalization()
        self.normalize9 = BatchNormalization()

        self.convolution1 = Conv2D(filters=256, kernel_size = (3, 3), padding="valid", input_shape=self.input_dim, kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.convolution2 = Conv2D(filters=256, kernel_size = (3, 3), padding="same",kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.convolution3 = Conv2D(filters=256, kernel_size = (3, 3), padding="same", kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.convolution4 = Conv2D(filters=256, kernel_size = (3, 3), padding="valid", kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.convolution5 = Conv2D(filters=256, kernel_size = (3, 3), padding="same", kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.convolution6 = Conv2D(filters=256, kernel_size = (3, 3), padding="same", kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.convolution7 = Conv2D(filters=256, kernel_size = (3, 3), padding="valid", kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.convolution8 = Conv2D(filters=256, kernel_size = (3, 3), padding="same", kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.convolution9 = Conv2D(filters=256, kernel_size = (3, 3), padding="same", kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))

        self.add = Add()
        self.relu = ReLU()
        self.policyHead = Dense(output_dimensions_policy, activation=softmax, kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))
        self.valueHead = Dense(1, activation=tanh, kernel_initializer='glorot_uniform',kernel_regularizer=l2(self.l2_const))

    def model(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x))

    def customLoss(self, y_true, y_pred):
        regressionLoss = reduce_sum((y_true[:, -1:] - y_pred[:, -1:]))
        print(regressionLoss)
        categoricalLoss = reduce_sum(binary_crossentropy(y_true[:, :-1], y_pred[:, :-1]))
        print(categoricalLoss)
        totalLoss = regressionLoss - categoricalLoss
        return totalLoss

    def call(self, inputs):
        # Residual layers using conv3d
        out = self.convolution1(inputs)
        out = self.normalize1(out)
        out = self.relu(out)

        out1 = self.convolution2(out)
        out1 = self.normalize2(out1)
        out1 = self.relu(out1)
        out1 = self.convolution3(out1)
        out1 = self.normalize3(out1)
        out = self.add([out, out1])
        out = self.relu(out)

        out = self.convolution4(out)
        out = self.normalize4(out)
        out = self.relu(out)

        out1 = self.convolution5(out)
        out1 = self.normalize5(out1)
        out1 = self.relu(out1)
        out1 = self.convolution6(out1)
        out1 = self.normalize6(out1)
        out = self.add([out, out1])
        out = self.relu(out)

        out = self.convolution7(out)
        out = self.normalize7(out)
        out = self.relu(out)
        out1 = self.convolution8(out)
        out1 = self.normalize8(out1)
        out1 = self.relu(out1)
        out1 = self.convolution9(out1)
        out1 = self.normalize9(out1)
        out = self.add([out, out1])
        out = self.relu(out)

        out = Flatten()(out)

        # Fully connected dense layers
        out = self.dense1(out)
        out = self.dense2(out)

        # Policy head and value head
        policyOut = self.policyHead(out)
        valueOut = self.valueHead(out)
        # final_out = concat([policyOut, valueOut], -1)
        return [policyOut, valueOut]


class NeuralNet:
    def __init__(self):
        self.current_model = models.load_model("base_model")
        # self.all_moves_ids = defaultdict(int)
        # self.ids_all_moves = defaultdict(Move)
        # c = 0
        # for i in range(64):
        #     for j in range(64):
        #         if i != j:
        #             self.all_moves_ids[Move(i, j)] = c
        #             self.ids_all_moves[c] = Move(i, j)
        #             c += 1

    def predict(self, input):
        return self.current_model.predict(input)

    def train(self, episodes):
        totalEpisodes = len(episodes)
        self.current_model.compile(loss=self.current_model.customLoss)
