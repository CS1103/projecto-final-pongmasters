//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include <cmath>

using namespace utec::neural_network;
   
template<typename T>
  class Dense final : public ILayer<T> {
  public:
  
    std::size_t in_f, out_f;
    Tensor<T,2> last_x;
    Tensor<T,2> last_z; 
    Tensor<T,2> w, b, dw, db;

    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) : in_f(in_f), out_f(out_f), w(in_f, out_f), b(1, out_f), dw(in_f, out_f), db(1, out_f){
      init_w_fun(w);
      init_b_fun(b);
    }

    Tensor<T,2> forward(const Tensor<T,2>& x) override {  // x -> 2 x 3 
      this->last_x = x;
      this->last_z = x*w+b;
      return last_z;
     }
     
    Tensor<T,2> backward(const Tensor<T,2>& dZ) override { 
      
      Tensor<T,2> dZ_mod = dZ;

    // Si se aplicó Sigmoid en forward, hay que derivarla
    if (out_f == 1) {
        for (size_t i = 0; i < last_z.data_.size(); ++i) {
            T sigmoid_val = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-last_z.data_[i]));
            dZ_mod.data_[i] = dZ.data_[i] * sigmoid_val * (1 - sigmoid_val);
        }
    }

    auto dx = dZ_mod * transpose_2d(w);
    dw = transpose_2d(last_x) * dZ_mod;

    db.fill(0);
    for(size_t i = 0; i < dZ.shape()[0]; ++i) {
        for(size_t j = 0; j < out_f; ++j){
            db(0,j) += dZ_mod(i,j);
        }
    }
      return dx;
     }

    void update_params(IOptimizer<T>& optimizer) override { 
      optimizer.update(w, dw);
      optimizer.update(b, db);
    }
  };



#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
