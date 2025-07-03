#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_dense.h"
#include "nn_optimizer.h"
#include <vector>
#include <memory>
#include <algorithm>

namespace utec::neural_network {

template<typename T>
class NeuralNetwork {
    std::vector<std::unique_ptr<ILayer<T>>> layers;

public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers.push_back(std::move(layer));
    }

    template <template <typename...> class LossType, 
              template <typename...> class Optimizer = SGD>
    void train(const Tensor<T,2>& X,
               const Tensor<T,2>& Y, 
               const size_t epochs, 
               const size_t batch_size, 
               T learning_rate,
               T decay_rate = 0.0) {

        Optimizer<T> optimizer(learning_rate);
    
        for (size_t e = 0; e < epochs; ++e) {
            if (decay_rate > 0) {
                T current_lr = learning_rate * (1.0 / (1.0 + decay_rate * e));
                optimizer.set_learning_rate(current_lr);
            }

            for (size_t i = 0; i < X.shape()[0]; i += batch_size) {
                size_t batch_len = std::min(batch_size, X.shape()[0] - i);
                
                Tensor<T,2> X_batch(batch_len, X.shape()[1]);
                Tensor<T,2> Y_batch(batch_len, Y.shape()[1]);

                for (size_t f = 0; f < batch_len; ++f) {
                    for (size_t c = 0; c < X.shape()[1]; ++c) {
                        X_batch(f, c) = X(i + f, c);
                    }
                    for (size_t c = 0; c < Y.shape()[1]; ++c) {
                        Y_batch(f, c) = Y(i + f, c);
                    }
                }

                auto y_pred = forward(X_batch);

                LossType<T> loss(y_pred, Y_batch);
                auto dloss = loss.loss_gradient();

                backward(dloss);

                for (auto& layer : layers) {
                    layer->update_params(optimizer);
                }
            }
            
            
            optimizer.step();
        }
    }

    Tensor<T,2> forward(const Tensor<T,2>& input) {
        Tensor<T,2> output = input;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    void backward(const Tensor<T,2>& gradients) {
        Tensor<T,2> grad = gradients;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
    }

    Tensor<T,2> predict(const Tensor<T,2>& X) {
        return forward(X);
    }
};

} // namespace utec::neural_network

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H