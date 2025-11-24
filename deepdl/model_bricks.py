import keras.optimizers
import scipy.linalg
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import numpy as np
from tensorflow.python.feature_column.feature_column_v2 import categorical_column_with_vocabulary_file
import matplotlib.pyplot as plt
import pickle
import tensorflow_probability as tfp
tfd =  tfp.distributions
tf.config.run_functions_eagerly(False)
import warnings



class PenaltyMatrix(tf.keras.Model):
    """
    Class used for construction of the penalty matrix for the complete additive model (of one Parameter). Idea is that
    individual penalties and their corresponding reg params are stored separately such that they can easily be accessed
    but also can be treated as a block-diagonal.
    """
    def __init__(self, matrices, reg_param, ti_penalties=None):
        super().__init__()
        self.reg_param = tf.Variable(initial_value=reg_param, trainable=False, dtype=tf.float32)
        self.pen_mats = [tf.constant(matrices[i], "float32") for i in range(len(matrices))]
        self.ti_penalties = ti_penalties
        if ti_penalties:
            self.ti_penalties = ti_penalties.copy()
            self.ti_len = len(ti_penalties)
            ti_penalties = [m for sublist in ti_penalties for m in sublist]
            ti_penalties = [tf.constant(ti_penalties[i], "float32") for i in range(len(ti_penalties))]
            self.pen_mats.extend(ti_penalties)
        self.smoother_idx = self.start_end()

    def matrix(self):
        pen_mats = [self.reg_param[i] * self.pen_mats[i] for i in range(len(self.pen_mats))]
        #if self.ti_penalties:
        #    mat1 = tf.linalg.LinearOperatorBlockDiag([tf.linalg.LinearOperatorFullMatrix(mat) for mat in pen_mats[:-1]]).to_dense()
        #    diff = (mat1.shape[0] - pen_mats[-1].shape[0])
        #    mat2 = tf.linalg.LinearOperatorBlockDiag([tf.linalg.LinearOperatorFullMatrix(tf.zeros((diff, diff))), tf.linalg.LinearOperatorFullMatrix(pen_mats[-1])]).to_dense()
        mat = tf.linalg.LinearOperatorBlockDiag(
                [tf.linalg.LinearOperatorFullMatrix(mat) for mat in pen_mats]).to_dense()
        if self.ti_penalties:
            k = self.ti_len  # number of special pairs
            normal, tis = pen_mats[:-2 * k], pen_mats[-2 * k:]
            pairs = [(tis[2 * i], tis[2 * i + 1]) for i in range(k)]

            mat1 = tf.linalg.LinearOperatorBlockDiag(
                [tf.linalg.LinearOperatorFullMatrix(m) for m in normal + [a for a, _ in pairs]]
            ).to_dense()

            zero_dim = sum(m.shape[0] for m in normal)
            mat2 = tf.linalg.LinearOperatorBlockDiag(
                ([tf.linalg.LinearOperatorFullMatrix(tf.zeros((zero_dim, zero_dim), "float32"))] if zero_dim else []) +
                [tf.linalg.LinearOperatorFullMatrix(b) for _, b in pairs]
            ).to_dense()
            return mat1, mat2
        return mat

    def matrices(self):
        return [self.reg_param[i] * self.pen_mats[i] for i in range(len(self.pen_mats))]




    def start_end(self):
        idxs = []
        dim = 0
        reduce = 0
        if self.ti_penalties:
            reduce = self.ti_len
        for i in range(len(self.pen_mats)-2*reduce):
            if i == 0:
                start = 0
            else:
                start = dim
            dim += self.pen_mats[i].shape[0]
            end = dim
            idxs.append((start, end))
        if reduce:
            for i in range(len(self.ti_penalties)):
                start = dim
                dim += self.ti_penalties[i][0].shape[0]
                end = dim
                idxs.append((start, end))
                idxs.append((start, end))
        return idxs


class RegressionModel(tf.keras.Model):
    def __init__(self, penalty, constant=False):
        super().__init__()
        self.linear = tf.keras.layers.Dense(1, use_bias=True, dtype="float32")
        self.penalty_matrix = penalty
        self.constant = constant

    @tf.function
    def call(self, inputs):
        return self.linear(inputs)

    @tf.function
    def penalty(self):
        if self.penalty_matrix.ti_penalties:
            mat1, mat2 = self.penalty_matrix.matrix()
            return tf.add(tf.matmul(tf.matmul(self.weights[0], mat1, transpose_a=True), self.weights[0]),
                          tf.matmul(tf.matmul(self.weights[0], mat2, transpose_a=True), self.weights[0]))

        return tf.matmul(tf.matmul(self.weights[0], self.penalty_matrix.matrix(), transpose_a=True), self.weights[0])




class SuperModel(tf.keras.Model):
    def __init__(self, models, N,categorical=False, dist="Normal"):
        super().__init__()
        self.models = [model for model in models]
        self.concat = tf.keras.layers.Concatenate()
        self.out = tf.keras.layers.Dense(1)
        self.categorical = categorical
        self.N = N
        self.dist = dist
    @tf.function
    def call(self, inputs):
        outs = self.linear_predictor(inputs)
        if self.dist == "Normal":
            return tfd.Normal(loc=outs[0], scale=tf.exp(outs[1]))
        if self.dist == "Beta":
            con0 =tf.math.softplus(outs[1])
            z = tf.math.sigmoid(outs[0])
            #con1 = (-z * con0) / (z - 1)
            alpha = z * con0
            beta = (1. - z) * con0

            return tfd.Beta(concentration1=alpha, concentration0=beta,
                                                        force_probs_to_zero_outside_support=False,
                                                       allow_nan_stats=False)

    @tf.function
    def to_dist(self, etas):
        if self.dist == "Normal":
            return tfd.Normal(loc=etas[0], scale=tf.exp(etas[1]))
        if self.dist == "Beta":
            con0 = tf.math.softplus(etas[1])
            z = tf.math.sigmoid(etas[0])
            #con1 = (-z * con0) / (z - 1)
            alpha = z * con0
            beta = (1. -z) * con0


            return tfd.Beta(concentration1=alpha, concentration0=beta,
                                                        force_probs_to_zero_outside_support=False,
                                                       allow_nan_stats=False)
    @tf.function
    def linear_predictor(self, inputs):
        x = [inp for inp in inputs]
        outs = [self.models[i](x[i]) for i in range(len(x))]
        if self.categorical:
            cat = outs.pop(0)
            outs[0] += cat[0]
            outs[1] += cat[1]
        return outs[0], outs[1]
    @tf.function
    def train_step(self, data):
        inputs, y = data
        batch_size = tf.cast(tf.shape(y)[0], "float32")
        with tf.GradientTape() as tape:
            y_pred = self(inputs)
            loss = self.compiled_loss(y, y_pred)
            start = 0
            if self.categorical:
                start = 1
            pen1 = self.models[start].penalty()
            pen2 = self.models[start+1].penalty()
            loss += tf.multiply(tf.cast(.5, "float32"), tf.multiply(tf.divide(batch_size, self.N), tf.add(pen1, pen2)))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return  {"loss": loss}



class LossLog_simpler(tf.keras.callbacks.Callback):
    def __init__(self, dataset, tolerance=10, outer_max=5, huge=False, out_name="smoothing_bigly", n_samples=10):
        super().__init__()
        self.epoch_losses = []
        self.tolerance = tolerance
        self.count = 0
        self.outer = 0
        self.outer_max = outer_max
        self.dataset = dataset
        self.old_pen_loss = 1e7
        self.huge = huge
        self.out_name = out_name
        self.n_samples = n_samples

  #  @tf.function
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.best_weights = self.model.weights
            self.initial_weights = self.model.weights
            self._edf1 = None
            self._edf2 = None
            self._lam1 = None
            self._lam2 = None
        loss = logs.get('loss')
        if loss is not None:
            self.epoch_losses.append(loss)

        if epoch > 3 and len(self.epoch_losses) > 2:
            if self.epoch_losses[-2] <= loss + 1e-7:
                self.count += 1
            else:
                self.count = 0


        if self.count >= self.tolerance and self.outer >= self.outer_max:
            self.model.stop_training = True

            #printvar = tf.print(self.model.reg_param)

        if self.count > self.tolerance and self.outer < self.outer_max and epoch < self.params["epochs"] - self.tolerance or self.huge:
            start = 0

            if self.model.categorical:
                start += 1
            smooth_dim1 = self.model.models[start].penalty_matrix.smoother_idx
            smooth_dim2 = self.model.models[start+1].penalty_matrix.smoother_idx
            edf1 = np.zeros((self.model.models[start].penalty_matrix.reg_param.shape[0]))
            edf2 = np.zeros((self.model.models[start+1].penalty_matrix.reg_param.shape[0]))
            pen_mats1 = self.model.models[start].penalty_matrix.matrices()
            pen_mats2 = self.model.models[start + 1].penalty_matrix.matrices()
            for batch in self.dataset.take(self.n_samples):
                x, y = batch
                with tf.GradientTape() as tape:
                    eta = self.model.linear_predictor(x)
                    y_pred = self.model.to_dist(eta)
                    loss = self.model.compiled_loss(y, y_pred)
                    grads = tape.gradient(loss, eta)
                start = 0
                if self.model.categorical:
                    start += 1
                    grads = [tf.reduce_sum(grads[i], axis=1) for i in range(len(grads))]
                xsubs1 = [x[start][:, smooth_dim1[i][0]:smooth_dim1[i][1]] for i in range(len(smooth_dim1))]
                xsubs2 = [x[start+1][:, smooth_dim1[i][0]:smooth_dim2[i][1]] for i in range(len(smooth_dim2))]




                XtX = [np.einsum('ni,n,nj->ij', X, np.square(grads[0].numpy().ravel()), X) for X in xsubs1]
                edf1 += np.array([np.trace(np.linalg.inv(XtX[i] + pen_mats1[i]) @ XtX[i]) for i in range(len(XtX))])
                XtX = [np.einsum('ni,n,nj->ij', X, np.square(grads[0].numpy().ravel()), X) for X in xsubs2]
                edf2 += np.array([np.trace(np.linalg.inv(XtX[i] + pen_mats2[i]) @ XtX[i]) for i in range(len(XtX))])

            edf1 /= self.n_samples
            edf2 /= self.n_samples

            lower1 = np.array([(tf.transpose(self.model.models[start].weights[0][smooth_dim1[i][0]:smooth_dim1[i][1]]) @ self.model.models[start].penalty_matrix.pen_mats[i] @ self.model.models[start].weights[0][smooth_dim1[i][0]:smooth_dim1[i][1]])[0] for i in range(len(self.model.models[start].penalty_matrix.pen_mats))]).flatten()
            lower2 = np.array([(tf.transpose(self.model.models[start+1].weights[0][smooth_dim2[i][0]:smooth_dim2[i][1]]) @ self.model.models[start+1].penalty_matrix.pen_mats[i] @ self.model.models[start+1].weights[0][smooth_dim2[i][0]:smooth_dim2[i][1]])[0] for i in range(len(self.model.models[start+1].penalty_matrix.pen_mats))]).flatten()
            if self.model.models[start].constant:
                lower1 = np.array([1.])
                edf1 = np.array([0.])
            if self.model.models[start+1].constant:
                lower2 = np.array([1.])
                edf2 = np.array([0.])
            if self.outer > 0:
                diffs = np.concatenate([self.edfs_old[0] - edf1,  self.edfs_old[1] - edf2])
                if np.all(np.abs(diffs) < 0.5) or np.any(edf1/lower1) < 0.:
                    self.edfs_old = [edf1, edf2]
                    self.model.stop_training = True
                else:
                    self.edfs_old = [edf1, edf2]
            else:
                self.edfs_old = [edf1, edf2]

            self.outer += 1
            self.model.models[start].penalty_matrix.reg_param.assign(edf1/lower1)
            self.model.models[start+1].penalty_matrix.reg_param.assign(edf2/lower2)
            self.count = 0
            self.epoch_losses = []


    def get_epoch_losses(self):
        return self.epoch_losses  # Return the list of epoch losses


def negloglik(y_true, y_pred):
    return -tf.reduce_sum(y_pred.log_prob(y_true))


